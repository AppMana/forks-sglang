"""
Ahead-of-time requantization for DeepSeek-V4-Flash to make it loadable on
Ampere consumer GPUs (sm_86: RTX 3090, RTX A5000).

V4-Flash storage format (verified empirically against the published checkpoint):
  - Routed-expert weights: MXFP4 (FP4 e2m1 nibbles packed as INT8 + per-32-element
    UE8M0 e8m0 scale). Shape: w1/w3 = (intermediate, hidden/2) INT8 +
    (intermediate, hidden/32) e8m0; w2 = (hidden, intermediate/2) INT8 +
    (hidden, intermediate/32) e8m0.
  - Attention weights (wq_a, wq_b, wkv, wo_a, wo_b): FP8 e4m3 + per-128x128-block
    UE8M0 e8m0 scale.
  - Norms / RoPE / shared experts / embed / head / hc-head / gate / attn_sink:
    BF16 / FP32 (no quantization).

Why we requantize: SGLang's existing MXFP4-Marlin path
(`mxfp4_deepseek.DeepSeekMxfp4MoEMethod` -> `prepare_moe_mxfp4_layer_for_marlin`)
hard-gates on `is_sm90_supported()` because the Marlin GEMM that consumes e8m0
scales requires Hopper FP8 tensor cores. Marlin INT4 with FP16 scales runs
natively on Ampere.

What we convert (matching the user's preference: keep precision tiers):
  - Routed-expert MXFP4 (4-bit weights + e8m0 scales)
        -> W4A16 (INT4 weights + FP16 group scales, group_size=32)
        -> SGLang's `gptq_marlin` MoE method on Ampere.
  - Attention FP8 e4m3 (8-bit weights + e8m0 scales, 128x128 blocks)
        -> W8A16 (INT8 weights + FP16 group scales, group_size=128)
        -> SGLang's `compressed_tensors`/`gptq_marlin` linear method.

Group-size strategy: we keep the existing block structure exactly. No
calibration corpus, no recalibration -- the per-block max we compute from the
FP4-dequanted FP32 values gives a tight FP16 scale that preserves the original
quantization granularity.

Shared experts, RoPE, norms, embed, head, attn_sink, hash-class head, gate,
and the V4-specific hc_attn_*/hc_ffn_* tensors are unchanged.

Public API:
  - `requantize_mxfp4_to_int4_w4a16(weight_packed_int8, scale_e8m0)` -> dict
  - `requantize_fp8_to_int8_w8a16(weight_fp8, scale_e8m0)` -> dict
  - `should_requantize_to_int4(name)`, `should_requantize_to_int8(name)` -- name
    classifiers used by the AOT shard converter.

Reuse from existing SGLang quantization utilities:
  - `mxfp4_tensor.MXFP4QuantizeUtil` (E2M1_values, dequantize)
  - `marlin_utils.marlin_permute_scales`, `marlin_utils.marlin_make_workspace`
  - `gptq.gptq_marlin_moe_repack` (downstream loader uses this)
"""
from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil


# E2M1 (FP4) levels in increasing absolute magnitude.
# Bit pattern (uint4): sign << 3 | magnitude
#   magnitude 0..7 -> {0, 0.5, 1, 1.5, 2, 3, 4, 6}
#   sign 0 -> +, sign 1 -> -
# Stored MXFP4: 2 nibbles per int8 byte, low 4 bits = even index, high 4 bits = odd.
_E2M1_VALUES = torch.tensor(MXFP4QuantizeUtil.E2M1_values, dtype=torch.float32)


def _unpack_int4_pairs(packed: torch.Tensor) -> torch.Tensor:
    """
    `packed` is an INT8/UINT8 tensor where each byte holds two 4-bit values:
      - low nibble  (bits 0-3) = even-indexed element
      - high nibble (bits 4-7) = odd-indexed element

    Returns a uint8 tensor with the last dim doubled, each entry in [0, 15].
    Mirrors the inverse of MXFP4QuantizeUtil's `fuse_uint4_to_uint8`.
    """
    # Treat the byte as unsigned regardless of original signedness.
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


def _e2m1_nibble_to_fp32(nibble: torch.Tensor) -> torch.Tensor:
    """
    Convert uint4 nibble (0..15) tensor to FP32 e2m1 values.
    sign = (nibble >> 3); magnitude = nibble & 0b111.
    """
    sign_bit = (nibble >> 3) & 1
    magnitude = (nibble & 0x07).to(torch.long)
    sign = 1.0 - 2.0 * sign_bit.to(torch.float32)  # 0 -> +1, 1 -> -1
    values = _E2M1_VALUES.to(nibble.device)
    mag_fp32 = values[magnitude.reshape(-1)].reshape(magnitude.shape)
    return sign * mag_fp32


def _e8m0_to_fp32_scale(scale_e8m0: torch.Tensor) -> torch.Tensor:
    """
    UE8M0 scale: stored as uint8 (or `torch.float8_e8m0fnu`), value = 2^(byte - 127).
    """
    if scale_e8m0.dtype == torch.float8_e8m0fnu:
        u = scale_e8m0.view(torch.uint8)
    elif scale_e8m0.dtype in (torch.int8, torch.uint8):
        u = scale_e8m0.view(torch.uint8) if scale_e8m0.dtype == torch.int8 else scale_e8m0
    else:
        raise TypeError(f"Unsupported e8m0 scale dtype: {scale_e8m0.dtype}")
    return torch.exp2(u.to(torch.float32) - 127.0)


def _pack_int4_pairs(unpacked: torch.Tensor) -> torch.Tensor:
    """
    Inverse of `_unpack_int4_pairs`. `unpacked` has values in [0, 15] (uint8).
    Returns a uint8 tensor with the last dim halved.

    Even indices go to low nibble, odd to high nibble, matching MXFP4's layout.
    Downstream Marlin repack will re-pack this however it needs; we keep the
    same byte order as MXFP4 storage so the conversion is a drop-in.
    """
    assert unpacked.shape[-1] % 2 == 0, f"last dim must be even, got {unpacked.shape}"
    low = unpacked[..., 0::2]
    high = unpacked[..., 1::2]
    return ((high & 0x0F) << 4) | (low & 0x0F)


def requantize_mxfp4_to_int4_w4a16(
    weight_packed_int8: torch.Tensor,
    scale_e8m0: torch.Tensor,
    *,
    out_scale_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    # BF16 (not FP16) is the right scale dtype on Ampere: BF16 has the same
    # 8-bit exponent as FP32, so any e8m0 value is representable in BF16
    # exactly (mantissa=0). FP16 has a 5-bit exponent and saturates at 2^±15,
    # which is too narrow for some V4-Flash blocks. BF16 also lets us encode
    # the *fractional* INT4 scales we need (no power-of-2 multiplier maps
    # FP4's {0.5, 1.5} levels onto INT4's integer grid).
    """
    Convert one MXFP4 expert weight tensor (packed FP4 + e8m0 scales) to W4A16
    (signed INT4 packed nibbles + FP16/BF16 per-group scales, group_size=32).

    Args:
      weight_packed_int8: shape (..., logical_K // 2), dtype int8 or uint8.
                          Each byte holds two FP4 e2m1 nibbles.
      scale_e8m0:         shape (..., logical_K // 32), dtype float8_e8m0fnu
                          (or uint8). One scale per 32 logical elements.
      out_scale_dtype:    BF16 (default) or FP16 -- target FP for new scales.

    Returns:
      dict with:
        "qweight_packed":  shape == weight_packed_int8.shape, dtype int8.
                           Two INT4 (signed, [-8..7]) nibbles per byte;
                           low nibble = even index, high = odd. Same byte
                           layout as the input MXFP4 storage.
        "scales":          shape == scale_e8m0.shape, dtype out_scale_dtype.
                           Per-32-element FP scale.
        "group_size":      32

    The conversion is per-block (32-element groups along the last logical dim):
      1. Dequant FP4 -> FP32 with e8m0 scale -> ground-truth values
      2. Symmetric per-block scale s = max(|values|) / 7
      3. INT4 (signed) = round(values / s).clamp(-8, 7)
      4. **Add 8 offset** to encode as UNSIGNED nibble [0, 15] for Marlin's
         GPTQ-compatible dequant (Marlin reads u4 and subtracts an implicit
         zero-point of 8 for symmetric quant when no zero-points tensor is
         provided -- equivalent to "value - 8" treated as signed).
      5. Re-pack low/high nibbles in MXFP4-compatible byte order.

    No calibration corpus is used; we keep the existing 32-element block
    structure and only switch from e8m0 (power-of-2) scales to BF16
    (full-precision) scales so the result loads via Marlin INT4 W4A16
    (`gptq_marlin`) on Ampere.
    """
    assert weight_packed_int8.dtype in (torch.int8, torch.uint8), (
        f"weight must be int8 or uint8, got {weight_packed_int8.dtype}"
    )
    device = weight_packed_int8.device

    # 1. Unpack FP4 nibbles and dequant to FP32.
    nibble = _unpack_int4_pairs(weight_packed_int8)        # (..., logical_K) uint8
    fp4_fp32 = _e2m1_nibble_to_fp32(nibble)                # (..., logical_K) FP32
    s_fp4 = _e8m0_to_fp32_scale(scale_e8m0)                # (..., logical_K // 32) FP32

    # Reshape last dim to (num_groups, group_size=32)
    last_dim = fp4_fp32.shape[-1]
    assert last_dim == s_fp4.shape[-1] * 32, (
        f"weight last dim {last_dim} != scale last dim {s_fp4.shape[-1]} * 32"
    )
    grouped = fp4_fp32.reshape(*fp4_fp32.shape[:-1], -1, 32)        # (..., G, 32)
    dequant = grouped * s_fp4.unsqueeze(-1)                         # (..., G, 32)

    # 2. Tight INT4 scale per group: max(|x|) / 7 (symmetric absmax).
    abs_max = dequant.abs().amax(dim=-1)                            # (..., G)
    abs_max = abs_max.clamp(min=torch.finfo(torch.float32).tiny)
    new_scale = abs_max / 7.0                                       # (..., G)

    # 3. Quantize to SIGNED INT4 [-8, 7], then offset by +8 to get UNSIGNED
    # nibble [0, 15]. Marlin treats the unsigned nibble as `(u4 - 8) * scale`
    # for symmetric quant.
    int4_signed = torch.round(dequant / new_scale.unsqueeze(-1))    # (..., G, 32)
    int4_signed = int4_signed.clamp(-8, 7)
    u4 = (int4_signed + 8).to(torch.uint8)                          # [0, 15]

    # 4. Re-pack: low nibble = even index, high nibble = odd. Same byte
    # order as MXFP4 storage so V4's existing weight_loader works unchanged.
    flat_u4 = u4.reshape(*fp4_fp32.shape)                           # (..., logical_K) uint8
    packed = _pack_int4_pairs(flat_u4)                              # (..., logical_K // 2) uint8
    qweight_packed = packed.view(torch.int8)                        # store as int8

    return {
        "qweight_packed": qweight_packed,
        "scales": new_scale.to(out_scale_dtype),
        "group_size": 32,
    }


def requantize_fp8_to_int8_w8a16(
    weight_fp8: torch.Tensor,
    scale_e8m0: torch.Tensor,
    *,
    block_size: tuple[int, int] = (128, 128),
    out_scale_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Convert one FP8-e4m3 attention weight tensor (with e8m0 per-128x128-block
    scales) to W8A16 (signed INT8 weights + FP16/BF16 per-block scales).

    Args:
      weight_fp8:    shape (N, K), dtype torch.float8_e4m3fn.
      scale_e8m0:    shape (ceil(N/128), ceil(K/128)), dtype float8_e8m0fnu.
      block_size:    (BN, BK), default (128, 128) per V4-Flash spec.
      out_scale_dtype: BF16 (default) or FP16 -- target FP for new scales.

    Returns:
      dict with:
        "qweight":     shape == weight_fp8.shape, dtype int8 (signed [-127..127]).
        "scales":      shape == scale_e8m0.shape, dtype out_scale_dtype.
        "block_size":  (BN, BK)

    Per-block conversion (BN x BK = 128 x 128):
      1. Dequant FP8 e4m3 -> FP32 using s_fp8 (e8m0)
      2. New INT8 scale s_int8 = max(|values|) / 127 (tight per block)
      3. INT8 = round(values / s_int8).clamp(-128, 127)
    """
    assert weight_fp8.dtype == torch.float8_e4m3fn, (
        f"weight must be float8_e4m3fn, got {weight_fp8.dtype}"
    )
    BN, BK = block_size
    N, K = weight_fp8.shape
    GN = (N + BN - 1) // BN
    GK = (K + BK - 1) // BK
    assert scale_e8m0.shape == (GN, GK), (
        f"expected scale shape ({GN}, {GK}), got {scale_e8m0.shape}"
    )

    # 1. Dequant FP8 -> FP32 with broadcasted e8m0 block scale.
    w_fp32 = weight_fp8.to(torch.float32)                           # (N, K)
    s_fp32 = _e8m0_to_fp32_scale(scale_e8m0)                        # (GN, GK)
    # Tile s_fp32 from (GN, GK) to (N, K) for elementwise multiply.
    s_full = s_fp32.repeat_interleave(BN, dim=0).repeat_interleave(BK, dim=1)
    s_full = s_full[:N, :K]                                          # crop padding
    dequant = w_fp32 * s_full                                       # (N, K)

    # 2. Per-block INT8 scale via tight absmax. Pad to (GN*BN, GK*BK) for
    #    block reduction, compute absmax over each block, divide by 127.
    pad_n = GN * BN - N
    pad_k = GK * BK - K
    padded = torch.nn.functional.pad(dequant, (0, pad_k, 0, pad_n))  # (GN*BN, GK*BK)
    blocked = padded.reshape(GN, BN, GK, BK).permute(0, 2, 1, 3)     # (GN, GK, BN, BK)
    abs_max = blocked.abs().amax(dim=(-2, -1))                       # (GN, GK)
    abs_max = abs_max.clamp(min=torch.finfo(torch.float32).tiny)
    new_scale = abs_max / 127.0                                      # (GN, GK)

    # 3. Quantize. Broadcast new_scale back to (N, K) and divide.
    scale_full = new_scale.repeat_interleave(BN, dim=0).repeat_interleave(BK, dim=1)
    scale_full = scale_full[:N, :K]
    qweight_fp32 = torch.round(dequant / scale_full)
    qweight = qweight_fp32.clamp(-128, 127).to(torch.int8)           # (N, K)

    return {
        "qweight": qweight,
        "scales": new_scale.to(out_scale_dtype),
        "block_size": (BN, BK),
    }


# ---------------------------------------------------------------------------
# Tensor-name classifiers (used by the AOT shard converter)
# ---------------------------------------------------------------------------

# Routed-expert weights live at e.g. `layers.0.ffn.experts.{0..255}.{w1,w2,w3}.weight`.
# Shared experts are NOT in this set (they're under `layers.N.ffn.shared_experts.*`
# with FP8 e4m3 dtype, treated as attention-class precision).
_EXPERT_WEIGHT_PATTERNS = (".ffn.experts.",)
_EXPERT_WEIGHT_LEAVES = (".w1.weight", ".w2.weight", ".w3.weight")
_EXPERT_SCALE_LEAVES = (".w1.scale", ".w2.scale", ".w3.scale")

# Attention FP8 weights live at e.g. `layers.0.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.weight`.
# Shared experts use the same FP8 path.
_FP8_WEIGHT_LEAVES = (".weight",)
_FP8_PARENT_PATTERNS = (
    ".attn.wq_a.",
    ".attn.wq_b.",
    ".attn.wkv.",
    ".attn.wo_a.",
    ".attn.wo_b.",
    # Indexer wq_b (only present on compressed layers, ratio in {4, 128})
    ".attn.indexer.wq_b.",
    ".ffn.shared_experts.w1.",
    ".ffn.shared_experts.w2.",
    ".ffn.shared_experts.w3.",
)


def dequantize_mxfp4_to_bf16(
    weight_packed_int8: torch.Tensor,
    scale_e8m0: torch.Tensor,
) -> torch.Tensor:
    """Pure dequantization: FP4 + e8m0 -> BF16 (no requantization). Used for
    proof-of-concept loading via SGLang's standard BF16 path. Memory cost:
    4x vs FP4 (4 bits -> 16 bits). Doesn't fit at chain scale; valid only
    for small / 1-layer tests."""
    nibble = _unpack_int4_pairs(weight_packed_int8)
    fp4 = _e2m1_nibble_to_fp32(nibble)
    s = _e8m0_to_fp32_scale(scale_e8m0)
    last = fp4.shape[-1]
    grouped = fp4.reshape(*fp4.shape[:-1], -1, 32) * s.unsqueeze(-1)
    return grouped.reshape(fp4.shape).to(torch.bfloat16)


def dequantize_fp8_to_bf16(
    weight_fp8: torch.Tensor,
    scale_e8m0: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """FP8 e4m3 + e8m0 (128x128 block) -> BF16."""
    BN, BK = block_size
    N, K = weight_fp8.shape
    s = _e8m0_to_fp32_scale(scale_e8m0)
    s_full = s.repeat_interleave(BN, dim=0).repeat_interleave(BK, dim=1)[:N, :K]
    return (weight_fp8.to(torch.float32) * s_full).to(torch.bfloat16)


def is_routed_expert_weight(name: str) -> bool:
    return any(p in name for p in _EXPERT_WEIGHT_PATTERNS) and name.endswith(
        _EXPERT_WEIGHT_LEAVES
    )


def is_routed_expert_scale(name: str) -> bool:
    return any(p in name for p in _EXPERT_WEIGHT_PATTERNS) and name.endswith(
        _EXPERT_SCALE_LEAVES
    )


def is_fp8_weight(name: str) -> bool:
    return any(p in name for p in _FP8_PARENT_PATTERNS) and name.endswith(".weight")


def is_fp8_scale(name: str) -> bool:
    return any(p in name for p in _FP8_PARENT_PATTERNS) and name.endswith(".scale")


def matched_scale_name(weight_name: str) -> Optional[str]:
    """Given a `*.weight` tensor name, return the corresponding `*.scale` name."""
    if weight_name.endswith(".weight"):
        return weight_name[: -len(".weight")] + ".scale"
    return None
