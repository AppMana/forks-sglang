"""
SGLang quantization config + methods for DeepSeek-V4-Flash that has been
AOT-requantized via `dsv4_aot_requantize.py`. This is the runtime side of
the requantization pipeline.

Storage format produced by the requant script:
  - Routed-expert weights:
      `*.ffn.experts.E.{w1,w2,w3}.weight`: int8, shape (out, in/2)
          two signed INT4 nibbles per byte (low=even, high=odd).
      `*.ffn.experts.E.{w1,w2,w3}.scale`:  bf16, shape (out, in/32)
          per-32-element symmetric absmax scales.
  - Attention / shared-expert weights:
      `*.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.weight`,
      `*.ffn.shared_experts.{w1,w2,w3}.weight`:
          int8, shape (N, K), signed INT8 [-128..127].
      Matching `.scale`: bf16, shape (ceil(N/128), ceil(K/128))
          per-128x128 block symmetric absmax scales.
  - Everything else (norms, rope, embed, head, hc_*, attn_sink, gate):
      passthrough (BF16 / FP32 unchanged).

Why this lives next to mxfp4.py:
  - We reuse Marlin INT4 W4A16 (sm_80+) for routed-experts. The path is the
    same as `prepare_moe_mxfp4_layer_for_marlin` minus the e8m0 epilogue.
  - We reuse a torch-native BF16 GEMM with per-block dequant for attention
    (simple and fast enough at low batch on Ampere; W8A16 Marlin is a
    later optimization).
  - The MXFP4 path's hard-coded `is_sm90_supported()` guard is gone here.

Activation: set `quantization_config.quant_method = "dsv4_int"` in the
checkpoint's config.json (the AOT requant script does this).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (shared between MoE method and Linear method)
# ---------------------------------------------------------------------------


def _unpack_int4_pairs(packed: torch.Tensor) -> torch.Tensor:
    """Unpack INT4 byte-pairs (low nibble = even index, high = odd) into a
    uint8 tensor with the last dim doubled. Mirrors the AOT pack_int4_pairs."""
    if packed.dtype == torch.int8:
        u = packed.view(torch.uint8)
    else:
        u = packed
    low = u & 0x0F
    high = (u >> 4) & 0x0F
    out_shape = list(u.shape)
    out_shape[-1] *= 2
    out = torch.empty(out_shape, dtype=torch.uint8, device=u.device)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


def _int4_signed(nibble: torch.Tensor) -> torch.Tensor:
    """Convert an unsigned int4 nibble in [0, 15] to its signed equivalent
    in [-8, 7] using the GPTQ/Marlin convention: signed = unsigned - 8.

    The AOT requantize stage stored values as `unsigned = signed + 8` so that
    Marlin's symmetric-quant kernel (which reads u4 and subtracts an implicit
    zero-point of 8) produces the right signed value at GEMM time. This
    function inverts that mapping for our own BF16 dequant path used in the
    process_weights_after_loading fallback / unit tests."""
    return nibble.to(torch.int8) - 8


def _dequant_int4_block_to_bf16(
    weight_packed_int8: torch.Tensor,
    scale_bf16: torch.Tensor,
    *,
    group_size: int = 32,
) -> torch.Tensor:
    """
    Dequantize byte-packed INT4 + per-32-element BF16 scales to BF16.

      weight: (..., in/2) int8
      scale:  (..., in/group_size) bf16
    Returns: (..., in) bf16
    """
    nibble = _unpack_int4_pairs(weight_packed_int8)        # (..., in) uint8
    int4 = _int4_signed(nibble).to(torch.float32)
    last = int4.shape[-1]
    grouped = int4.reshape(*int4.shape[:-1], -1, group_size)
    out = grouped * scale_bf16.to(torch.float32).unsqueeze(-1)
    return out.reshape(*int4.shape[:-1], last).to(torch.bfloat16)


def _dequant_int8_block_to_bf16(
    weight_int8: torch.Tensor,
    scale_bf16: torch.Tensor,
    *,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """
    Dequantize INT8 + per-(BN, BK)-block BF16 scales to BF16.

      weight: (N, K) int8
      scale:  (ceil(N/BN), ceil(K/BK)) bf16
    Returns: (N, K) bf16
    """
    N, K = weight_int8.shape
    BN, BK = block_size
    s = scale_bf16.to(torch.float32)
    s_full = s.repeat_interleave(BN, dim=0).repeat_interleave(BK, dim=1)[:N, :K]
    return (weight_int8.to(torch.float32) * s_full).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# QuantizationConfig
# ---------------------------------------------------------------------------


class Dsv4IntConfig(QuantizationConfig):
    """V4-Flash AOT-INT-quantized config. Routes routed-experts through
    `Dsv4Int4MoEMethod` (W4A16) and attention/shared-experts through
    `Dsv4Int8LinearMethod` (W8A16). Anything not matching either set is
    treated as unquantized BF16.
    """

    QUANT_METHOD_NAME = "dsv4_int"

    # Tensor-name patterns that route to the INT4 MoE method.
    EXPERT_PARENT_PATTERNS = (".ffn.experts.",)
    EXPERT_LEAVES = ("w1", "w2", "w3")

    # Tensor-name patterns that route to the INT8 Linear method.
    # Note these patterns match the *runtime* layer prefix (after V4's
    # remap_weight_name_to_dpsk_hf_format), which uses `.self_attn.` and
    # `.mlp.shared_experts.` (not `.attn.` / `.ffn.shared_experts.`).
    INT8_PARENT_PATTERNS = (
        ".self_attn.wq_a",
        ".self_attn.wq_b",
        ".self_attn.wkv",
        ".self_attn.wo_a",
        ".self_attn.wo_b",
        # Compressed layers (compress_ratio in {4, 128}) have an extra
        # indexer.wq_b linear that's also FP8 in the source.
        ".self_attn.indexer.wq_b",
        ".mlp.shared_experts.w1",
        ".mlp.shared_experts.w2",
        ".mlp.shared_experts.w3",
        # Shared experts may be fused into gate_up_proj / down_proj after
        # SGLang's MoE fusion; cover those too.
        ".mlp.shared_experts.gate_proj",
        ".mlp.shared_experts.up_proj",
        ".mlp.shared_experts.down_proj",
        ".mlp.shared_experts.gate_up_proj",
    )

    def __init__(
        self,
        config_groups: Optional[Dict[str, Any]] = None,
        ignore_patterns: Optional[List[str]] = None,
    ):
        super().__init__()
        self.config_groups = config_groups or {}
        self.ignore_patterns = ignore_patterns or []
        # Mirror Fp8Config's interface: linear.weight_loader_v2 reads
        # `quant_method.quant_config.weight_block_size` for block scales.
        self.weight_block_size = (128, 128)
        self.activation_scheme = "dynamic"

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Dsv4IntConfig":
        return cls(
            config_groups=config.get("config_groups", {}),
            ignore_patterns=config.get("ignore", []),
        )

    @classmethod
    def get_name(cls) -> str:
        return cls.QUANT_METHOD_NAME

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80  # Marlin INT4 W4A16

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        if isinstance(layer, FusedMoE):
            return Dsv4Int4MoEMethod(prefix=prefix, quant_config=self)
        if isinstance(layer, LinearBase):
            if any(p in prefix for p in self.INT8_PARENT_PATTERNS):
                return Dsv4Int8LinearMethod(quant_config=self)
            # gate.weight (router), shared norms, rope tables, etc. are BF16.
            return UnquantizedLinearMethod()
        return None


# ---------------------------------------------------------------------------
# MoE method (routed-experts: INT4 byte-packed + BF16 group scales)
# ---------------------------------------------------------------------------


class Dsv4Int4MoEMethod(FusedMoEMethodBase):
    """
    Loads the routed-expert weights produced by the AOT requant script
    and runs them through Marlin INT4 W4A16 on Ampere (sm_80+).

    Storage layout (per the AOT script):
      - per-expert `w1`, `w2`, `w3` tensors are int8 byte-packed INT4 nibbles
      - per-expert scale tensors are bf16 (out, in/32)

    SGLang's FusedMoE layer fuses per-expert w1+w3 into `w13_weight` and keeps
    w2 separate. Our `create_weights` follows that convention.

    On `process_weights_after_loading` we call `gptq_marlin_repack` to convert
    the byte-packed INT4 weights into Marlin's int32 layout, then permute the
    BF16 scales for Marlin's expected layout. The Marlin GEMM kernel then
    runs on Ampere natively (sm_80+, no e8m0 epilogue, no Hopper FP8).
    """

    GROUP_SIZE = 32

    def __init__(self, prefix: str = "", quant_config: Optional["Dsv4IntConfig"] = None):
        self.prefix = prefix
        self.quant_config = quant_config

    # FusedMoEMethodBase API ----------------------------------------------

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.linear import set_weight_attrs
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size_per_partition
        self.params_dtype = params_dtype
        self.with_bias = with_bias

        # Fused gate_up_proj: w13 = [w1; w3] along intermediate dim
        # logical shape (num_experts, 2*intermediate, hidden); stored as int8
        # byte-packed: (num_experts, 2*intermediate, hidden//2)
        w13_weight = Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale_inv = Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.GROUP_SIZE,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale_inv)
        set_weight_attrs(w13_weight_scale_inv, {
            "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
            **extra_weight_attrs,
        })

        # down_proj: w2 (num_experts, hidden, intermediate); stored
        # byte-packed: (num_experts, hidden, intermediate//2)
        w2_weight = Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale_inv = Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.GROUP_SIZE,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale_inv)
        set_weight_attrs(w2_weight_scale_inv, {
            "quant_method": FusedMoeWeightScaleSupported.GROUP.value,
            **extra_weight_attrs,
        })

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert byte-packed INT4 to Marlin int32 layout in place."""
        from sgl_kernel import gptq_marlin_repack
        from sglang.srt.layers.quantization.marlin_utils import (
            marlin_make_workspace,
            marlin_permute_scales,
        )

        device = layer.w13_weight.device
        layer.workspace = marlin_make_workspace(device, 4)
        perm = torch.empty(0, dtype=torch.int, device=device)

        num_experts = self.num_experts
        hidden_size = self.hidden_size
        intermediate = self.intermediate_size_per_partition
        group_size = self.GROUP_SIZE

        def _repack(weight_int8_packed: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
            """Pack INT4 nibbles for Marlin: (E, N, K/2 int8) -> Marlin layout.
            Pre-allocates the output to avoid the 2x memory peak from
            torch.stack() concatenation."""
            # Probe one expert to get the output shape.
            w0 = weight_int8_packed[0].view(torch.uint8).view(torch.int32).T.contiguous()
            probe = gptq_marlin_repack(
                b_q_weight=w0, perm=perm, size_k=size_k, size_n=size_n, num_bits=4,
            )
            out = torch.empty(
                num_experts, *probe.shape, dtype=probe.dtype, device=probe.device,
            )
            out[0] = probe
            del probe
            for e in range(1, num_experts):
                w = weight_int8_packed[e].view(torch.uint8).view(torch.int32).T.contiguous()
                marlin_qweight = gptq_marlin_repack(
                    b_q_weight=w, perm=perm, size_k=size_k, size_n=size_n, num_bits=4,
                )
                out[e].copy_(marlin_qweight)
                del marlin_qweight
            return out

        def _permute_scales(scales_bf16: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
            """Marlin expects scales transposed + permuted. Pre-allocates."""
            s0 = scales_bf16[0].T.contiguous()
            probe = marlin_permute_scales(
                s=s0, size_k=size_k, size_n=size_n, group_size=group_size
            )
            out = torch.empty(
                num_experts, *probe.shape, dtype=probe.dtype, device=probe.device,
            )
            out[0] = probe
            del probe
            for e in range(1, num_experts):
                s = scales_bf16[e].T.contiguous()
                m = marlin_permute_scales(
                    s=s, size_k=size_k, size_n=size_n, group_size=group_size
                )
                out[e].copy_(m)
                del m
            return out

        # Process one parameter at a time, freeing the original before
        # allocating the Marlin output to keep peak memory below 24 GB.

        # w13: size_n = 2*intermediate, size_k = hidden
        w13_marlin = _repack(layer.w13_weight.data, 2 * intermediate, hidden_size)
        layer.w13_weight = Parameter(w13_marlin, requires_grad=False)
        del w13_marlin
        torch.cuda.empty_cache()

        w13_scales_marlin = _permute_scales(layer.w13_weight_scale_inv.data,
                                            2 * intermediate, hidden_size)
        layer.w13_weight_scale_inv = Parameter(w13_scales_marlin, requires_grad=False)
        del w13_scales_marlin
        torch.cuda.empty_cache()

        # w2: size_n = hidden, size_k = intermediate
        w2_marlin = _repack(layer.w2_weight.data, hidden_size, intermediate)
        layer.w2_weight = Parameter(w2_marlin, requires_grad=False)
        del w2_marlin
        torch.cuda.empty_cache()

        w2_scales_marlin = _permute_scales(layer.w2_weight_scale_inv.data,
                                           hidden_size, intermediate)
        layer.w2_weight_scale_inv = Parameter(w2_scales_marlin, requires_grad=False)
        del w2_scales_marlin
        torch.cuda.empty_cache()

        layer._dsv4_int_marlin_ready = True

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config) -> None:
        # Save runner config but use direct fused_marlin_moe call below.
        # The MoeRunner.MARLIN abstraction was breaking dispatch in V4's
        # forward path; the direct call is the path validated standalone.
        self.moe_runner_config = moe_runner_config

    def apply(self, layer, dispatch_output):
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids
        router_logits = getattr(topk_output, "router_logits", None)

        if not hasattr(layer, "_dsv4_marlin_workspace"):
            layer._dsv4_marlin_workspace = marlin_make_workspace(
                x.device, max_blocks_per_sm=4
            )

        device = x.device
        empty_g_idx = torch.empty(self.num_experts, 0, dtype=torch.int32, device=device)
        out = fused_marlin_moe(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale_inv,
            w2_scale=layer.w2_weight_scale_inv,
            gating_output=router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            g_idx1=empty_g_idx,
            g_idx2=empty_g_idx,
            sort_indices1=empty_g_idx,
            sort_indices2=empty_g_idx,
            num_bits=4,
            workspace=layer._dsv4_marlin_workspace,
            is_k_full=True,
        )
        return StandardCombineInput(hidden_states=out)


# ---------------------------------------------------------------------------
# Linear method (attention + shared experts: INT8 + BF16 block scales)
# ---------------------------------------------------------------------------


class Dsv4Int8LinearMethod(LinearMethodBase):
    """
    Loads INT8 weights with BF16 per-128x128-block scales and runs a
    BF16 matmul with per-block dequantization.

    For low to moderate batch sizes on Ampere this is competitive with
    Marlin INT8 W8A16; for very high throughput we'd switch to Marlin
    later. Simplicity wins here -- we don't have to repack scales into
    Marlin's group layout (which assumes 1D groups along K, not 2D blocks).
    """

    BLOCK_SIZE = (128, 128)

    def __init__(self, quant_config: Optional["Dsv4IntConfig"] = None):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.linear import set_weight_attrs

        out_dim = sum(output_partition_sizes)
        in_dim = input_size_per_partition
        BN, BK = self.BLOCK_SIZE
        gn = (out_dim + BN - 1) // BN
        gk = (in_dim + BK - 1) // BK

        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = ModelWeightParameter(
            data=torch.zeros(out_dim, in_dim, dtype=torch.int8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # V4-Flash code asserts wo_a has `weight_scale_inv` (FP8 naming
        # convention). BlockQuantScaleParameter handles per-block sharding
        # correctly given output_dim/input_dim; without it the merged-column
        # loader would try to slice along axis 0 with full out_dim instead
        # of out_dim/block_n.
        weight_scale = BlockQuantScaleParameter(
            data=torch.zeros(gn, gk, dtype=torch.bfloat16),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale_inv", weight_scale)

        layer._dsv4_int_block_size = self.BLOCK_SIZE

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Dequantize INT8 -> BF16 once (memory cost: 2x per attention weight,
        # acceptable -- attention is ~5% of total weights, dominated by experts).
        # This lets V4's hardcoded `wo_a.weight.view(...)` einsum and other
        # raw-weight-access paths continue to work without modification.
        BN, BK = self.BLOCK_SIZE
        w_bf16 = _dequant_int8_block_to_bf16(
            layer.weight.data, layer.weight_scale_inv.data, block_size=(BN, BK)
        )
        layer.weight = Parameter(w_bf16.contiguous(), requires_grad=False)
        # Keep weight_scale_inv attribute alive (V4 model code accesses it
        # directly), but it's no longer used by our apply().
        layer._dsv4_int_dequanted = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Weights are already BF16 after process_weights_after_loading.
        return torch.nn.functional.linear(x, layer.weight, bias)
