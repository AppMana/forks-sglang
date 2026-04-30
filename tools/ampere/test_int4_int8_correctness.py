"""
End-to-end numerical correctness test for the INT4/INT8 AOT requantization
+ Marlin GEMM path on Ampere sm_86, using REAL V4-Flash expert weights from
the converted /var/lib/inference/v4-flash-1layer-int dir.

We validate three things:

  1. AOT requant round-trip:
        FP4_dequant_groundtruth ≈ INT4_dequant(after_requant)
        FP8_dequant_groundtruth ≈ INT8_dequant(after_requant)

  2. Marlin INT4 W4A16 GEMM produces output that matches a BF16 groundtruth
     GEMM (using INT4-dequantized weights as the BF16 reference).

  3. The full per-expert MoE math (gate*up activation -> down) on a single
     expert matches a BF16-only reference within tolerance.

This isolates the kernel/math correctness from the SGLang server pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from safetensors import safe_open
from sgl_kernel import gptq_marlin_repack
from sglang.srt.layers.quantization.marlin_utils import (
    marlin_make_workspace,
    marlin_permute_scales,
)
from sglang.srt.layers.quantization.dsv4_aot_requantize import (
    _e2m1_nibble_to_fp32,
    _e8m0_to_fp32_scale,
    _unpack_int4_pairs,
)
from sglang.srt.layers.quantization.dsv4_int import (
    _dequant_int4_block_to_bf16,
    _dequant_int8_block_to_bf16,
)


SRC_REQUANT = Path("/var/lib/inference/v4-flash-1layer-int")
ORIGINAL_FP4 = Path(
    "/home/administrator/.cache/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-V4-Flash/snapshots/"
    "fd53f944496234770ba80e15004f9b6d269a71f5"
)


def log(msg: str):
    print(f"[test] {msg}", flush=True)


def fp4_dequant_groundtruth(weight_packed_int8, scale_e8m0):
    """Reference dequant (matches MXFP4QuantizeUtil.dequantize)."""
    nibble = _unpack_int4_pairs(weight_packed_int8)
    fp32 = _e2m1_nibble_to_fp32(nibble)
    s = _e8m0_to_fp32_scale(scale_e8m0)
    grouped = fp32.reshape(*fp32.shape[:-1], -1, 32)
    return (grouped * s.unsqueeze(-1)).reshape(fp32.shape)


def fp8_dequant_groundtruth(weight_fp8, scale_e8m0, block=(128, 128)):
    """Reference FP8 e4m3 + e8m0 (block) -> FP32."""
    BN, BK = block
    N, K = weight_fp8.shape
    s = _e8m0_to_fp32_scale(scale_e8m0)
    s_full = s.repeat_interleave(BN, dim=0).repeat_interleave(BK, dim=1)[:N, :K]
    return weight_fp8.to(torch.float32) * s_full


def test_requant_roundtrip(device: str):
    """Verify INT4 / INT8 requantized weights match the FP4 / FP8 ground truth
    after dequantization. Lossy by design but the SNR should be acceptable."""
    log("=== Test 1: AOT requant round-trip vs source FP4/FP8 ===")

    # Load ORIGINAL FP4 expert + INT4 requantized expert and compare.
    src = f"{ORIGINAL_FP4}/model-00002-of-00046.safetensors"
    dst = f"{SRC_REQUANT}/model-00002-of-00046.safetensors"

    with safe_open(src, framework="pt", device=device) as fs, \
         safe_open(dst, framework="pt", device=device) as fd:
        # Pick a sample expert weight
        w_fp4 = fs.get_tensor("layers.0.ffn.experts.0.w1.weight")
        s_fp4 = fs.get_tensor("layers.0.ffn.experts.0.w1.scale")
        w_int4 = fd.get_tensor("layers.0.ffn.experts.0.w1.weight")
        s_int4 = fd.get_tensor("layers.0.ffn.experts.0.w1.scale")

        log(f"  fp4 weight {tuple(w_fp4.shape)} {w_fp4.dtype}, scale {tuple(s_fp4.shape)} {s_fp4.dtype}")
        log(f"  int4 weight {tuple(w_int4.shape)} {w_int4.dtype}, scale {tuple(s_int4.shape)} {s_int4.dtype}")

        fp4_truth = fp4_dequant_groundtruth(w_fp4, s_fp4)
        int4_dequant = _dequant_int4_block_to_bf16(w_int4, s_int4, group_size=32).to(torch.float32)

        err = (fp4_truth - int4_dequant).abs()
        snr = 20 * torch.log10(fp4_truth.norm() / (fp4_truth - int4_dequant).norm()).item()
        log(f"  Expert w1: SNR={snr:.2f} dB, max_err={err.max().item():.4g}, mean_err={err.mean().item():.4g}")
        assert snr > 17, f"Expert SNR too low: {snr:.2f} dB"

        # Same for FP8 attention
        w_fp8 = fs.get_tensor("layers.0.attn.wq_a.weight")
        s_fp8 = fs.get_tensor("layers.0.attn.wq_a.scale")
        w_int8 = fd.get_tensor("layers.0.attn.wq_a.weight")
        s_int8 = fd.get_tensor("layers.0.attn.wq_a.scale")

        log(f"  fp8 attn weight {tuple(w_fp8.shape)} {w_fp8.dtype}, scale {tuple(s_fp8.shape)} {s_fp8.dtype}")
        log(f"  int8 attn weight {tuple(w_int8.shape)} {w_int8.dtype}, scale {tuple(s_int8.shape)} {s_int8.dtype}")

        fp8_truth = fp8_dequant_groundtruth(w_fp8, s_fp8)
        int8_dequant = _dequant_int8_block_to_bf16(w_int8, s_int8, block_size=(128, 128)).to(torch.float32)
        snr8 = 20 * torch.log10(fp8_truth.norm() / (fp8_truth - int8_dequant).norm()).item()
        err8 = (fp8_truth - int8_dequant).abs()
        log(f"  Attn wq_a: SNR={snr8:.2f} dB, max_err={err8.max().item():.4g}, mean_err={err8.mean().item():.4g}")
        assert snr8 > 35, f"Attn SNR too low: {snr8:.2f} dB"
    log("  PASS: requant round-trip within tolerance")


def test_marlin_gemm_correctness(device: str):
    """Verify Marlin INT4 W4A16 fused-MoE GEMM matches a BF16 reference for
    a single expert. This mirrors what Dsv4Int4MoEMethod produces at load
    time and apply time."""
    log("=== Test 2: Marlin INT4 W4A16 fused-MoE == BF16 reference ===")

    HIDDEN = 4096
    INTER = 2048
    GROUP = 32

    dst = f"{SRC_REQUANT}/model-00002-of-00046.safetensors"

    # Build a single-expert fused w13 + w2 from the requantized weights.
    with safe_open(dst, framework="pt", device=device) as f:
        w1 = f.get_tensor("layers.0.ffn.experts.0.w1.weight")  # int8 (INTER=2048, HIDDEN/2=2048)
        w3 = f.get_tensor("layers.0.ffn.experts.0.w3.weight")  # same
        s1 = f.get_tensor("layers.0.ffn.experts.0.w1.scale")   # bf16 (INTER, HIDDEN/GROUP=128)
        s3 = f.get_tensor("layers.0.ffn.experts.0.w3.scale")
        w2 = f.get_tensor("layers.0.ffn.experts.0.w2.weight")  # int8 (HIDDEN, INTER/2)
        s2 = f.get_tensor("layers.0.ffn.experts.0.w2.scale")   # bf16 (HIDDEN, INTER/GROUP=64)

    # Fuse w13 = [w1 ; w3] along output (intermediate) dim
    w13 = torch.cat([w1, w3], dim=0)         # (2*INTER, HIDDEN/2) = (4096, 2048) int8
    s13 = torch.cat([s1, s3], dim=0)         # (2*INTER, HIDDEN/GROUP) = (4096, 128) bf16
    log(f"  w13 fused: {tuple(w13.shape)} {w13.dtype}, s13 {tuple(s13.shape)} {s13.dtype}")

    perm = torch.empty(0, dtype=torch.int, device=device)

    # Marlin repack w13 (size_n=2*INTER=4096, size_k=HIDDEN=4096)
    qw13 = w13.view(torch.uint8).view(torch.int32).T.contiguous()  # (HIDDEN/8=512, 2*INTER=4096)
    marlin_w13 = gptq_marlin_repack(
        b_q_weight=qw13, perm=perm, size_k=HIDDEN, size_n=2 * INTER, num_bits=4,
    )
    s13_t = s13.T.contiguous()  # (HIDDEN/GROUP, 2*INTER) = (128, 4096)
    marlin_s13 = marlin_permute_scales(
        s=s13_t, size_k=HIDDEN, size_n=2 * INTER, group_size=GROUP
    )
    log(f"  marlin_w13 {tuple(marlin_w13.shape)} {marlin_w13.dtype}, marlin_s13 {tuple(marlin_s13.shape)} {marlin_s13.dtype}")

    # Marlin repack w2 (size_n=HIDDEN=4096, size_k=INTER=2048)
    qw2 = w2.view(torch.uint8).view(torch.int32).T.contiguous()
    marlin_w2 = gptq_marlin_repack(
        b_q_weight=qw2, perm=perm, size_k=INTER, size_n=HIDDEN, num_bits=4,
    )
    s2_t = s2.T.contiguous()
    marlin_s2 = marlin_permute_scales(
        s=s2_t, size_k=INTER, size_n=HIDDEN, group_size=GROUP
    )
    log(f"  marlin_w2 {tuple(marlin_w2.shape)} {marlin_w2.dtype}, marlin_s2 {tuple(marlin_s2.shape)} {marlin_s2.dtype}")

    # --- BF16 ground-truth path: dequant fused INT4 -> BF16, run silu(w1.x) * (w3.x), w2 ---
    # w1, w3 dequanted separately; w2 dequanted; full SiLU MoE math.
    bf16_w1 = _dequant_int4_block_to_bf16(w1, s1, group_size=GROUP)  # (INTER, HIDDEN)
    bf16_w3 = _dequant_int4_block_to_bf16(w3, s3, group_size=GROUP)
    bf16_w2 = _dequant_int4_block_to_bf16(w2, s2, group_size=GROUP)  # (HIDDEN, INTER)

    M = 8
    x = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1
    log(f"  x: {tuple(x.shape)} {x.dtype} std={x.std().item():.4g}")

    # ref: silu(x @ w1.T) * (x @ w3.T) @ w2.T
    gate_ref = torch.nn.functional.silu(x @ bf16_w1.T)
    up_ref = x @ bf16_w3.T
    inter_ref = gate_ref * up_ref               # (M, INTER)
    out_bf16_ref = inter_ref @ bf16_w2.T        # (M, HIDDEN)
    log(f"  bf16_ref output: {tuple(out_bf16_ref.shape)} std={out_bf16_ref.std().item():.4g}")

    # --- Marlin path through fused_marlin_moe with 1 expert, top_k=1 ---
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe

    w13_q = marlin_w13.unsqueeze(0)
    w13_s = marlin_s13.unsqueeze(0)
    w2_q = marlin_w2.unsqueeze(0)
    w2_s = marlin_s2.unsqueeze(0)

    workspace = marlin_make_workspace(device, max_blocks_per_sm=4)
    empty_g = torch.empty(1, 0, dtype=torch.int32, device=device)
    topk_w = torch.ones(M, 1, dtype=torch.float32, device=device)
    topk_ids = torch.zeros(M, 1, dtype=torch.int32, device=device)
    gating = torch.zeros(M, 1, dtype=torch.bfloat16, device=device)

    out_marlin = fused_marlin_moe(
        hidden_states=x.contiguous(),
        w1=w13_q, w2=w2_q,
        w1_scale=w13_s, w2_scale=w2_s,
        gating_output=gating, topk_weights=topk_w, topk_ids=topk_ids,
        g_idx1=empty_g, g_idx2=empty_g,
        sort_indices1=empty_g, sort_indices2=empty_g,
        num_bits=4, workspace=workspace, is_k_full=True,
    )
    log(f"  marlin output: {tuple(out_marlin.shape)} {out_marlin.dtype} std={out_marlin.std().item():.4g}")

    # Compare numerically
    diff = (out_marlin.float() - out_bf16_ref.float())
    rel = diff.norm() / out_bf16_ref.float().norm()
    snr = -20 * torch.log10(rel + 1e-30).item()
    log(f"  Marlin vs BF16 ref: rel={rel.item():.4g}, SNR={snr:.2f} dB")
    log(f"  max_abs_err={diff.abs().max().item():.4g}, mean_abs_err={diff.abs().mean().item():.4g}")
    log(f"  marlin sample: {out_marlin[0, :5].tolist()}")
    log(f"  bf16_ref sample: {out_bf16_ref[0, :5].tolist()}")

    assert torch.isfinite(out_marlin).all(), "Marlin output has NaN/Inf"
    assert snr > 20, f"Marlin numerical SNR too low: {snr:.2f} dB (expected > 20 dB)"
    log(f"  PASS: Marlin matches BF16 reference within {snr:.1f} dB")


def main():
    torch.cuda.set_device(1)
    device = "cuda:1"
    log(f"GPU: {torch.cuda.get_device_name(1)} cap={torch.cuda.get_device_capability(1)}")
    test_requant_roundtrip(device)
    test_marlin_gemm_correctness(device)
    log("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
