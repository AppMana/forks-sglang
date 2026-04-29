"""
Local Ampere (sm_86) shape/correctness probe for SGLang's TileLang NSA kernels.

This is the gate before any cluster work: validate that TileLang JIT compiles
and runs on Ampere for the kernels DeepSeek-V4-Flash will use. Garbage output
is fine -- we want shape correctness and successful kernel launch.

Run with:
    cd ~/Documents/appmana/forks-sglang
    source .venv/bin/activate
    python tools/ampere/test_tilelang_nsa.py
"""
from __future__ import annotations

import os
import sys

import torch

assert torch.cuda.is_available(), "CUDA not available"
DEVICE = torch.device("cuda")
DEV_CC = torch.cuda.get_device_capability(0)
print(f"[probe] device={torch.cuda.get_device_name(0)} cc={DEV_CC[0]}.{DEV_CC[1]}")
assert DEV_CC[0] == 8, f"this probe expects sm_8x (Ampere/Ada); got cc={DEV_CC}"

# Quiet noisy import-time logs.
os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")
os.environ.setdefault("SGLANG_DSV4_FP4_EXPERTS", "0")


def heading(name: str) -> None:
    print(f"\n=== {name} ===")


def assert_shape(actual: torch.Tensor, expected: tuple[int, ...], label: str) -> None:
    if tuple(actual.shape) != tuple(expected):
        raise AssertionError(f"{label}: got shape {tuple(actual.shape)}, expected {expected}")
    print(f"  {label}: shape={tuple(actual.shape)} dtype={actual.dtype} OK")


# ----- 1. act_quant_kernel: BF16 -> FP8 + per-block float32 scales --------
heading("act_quant_kernel (BF16 -> FP8E4M3 with block scales)")
from sglang.srt.layers.attention.nsa.tilelang_kernel import act_quant

bs, seq, dim = 2, 64, 4096
block_size = 128
x = torch.randn(bs, seq, dim, dtype=torch.bfloat16, device=DEVICE)
y, s = act_quant(x, block_size=block_size)
assert_shape(y, (bs, seq, dim), "y (quantized)")
assert_shape(s, (bs, seq, dim // block_size), "s (scales)")
assert y.dtype == torch.float8_e4m3fn, f"y dtype {y.dtype}"
assert s.dtype == torch.float32, f"s dtype {s.dtype}"

# ----- 2. fp8_index: dispatcher picks Ampere BF16 path on sm_86 -----------
heading("fp8_index (dispatcher: Ampere BF16 fallback for sm<89)")
from sglang.srt.layers.attention.nsa.tilelang_kernel import (
    fp8_index,
    _device_has_fp8_mma,
)

# Same shapes the V4 indexer uses: q[b, m, h, d] FP8, q_s[b, m, h] FP32,
# k[b, n, d] FP8, k_s[b, n] FP32 -> o[b, m, n] FP32. Build directly so the
# scales align with the kernel's expected shape.
B, M, H, D, N = 1, 16, 64, 128, 32
q = torch.randn(B, M, H, D, device=DEVICE).to(torch.float8_e4m3fn)
q_s = torch.rand(B, M, H, device=DEVICE, dtype=torch.float32)
k = torch.randn(B, N, D, device=DEVICE).to(torch.float8_e4m3fn)
k_s = torch.rand(B, N, device=DEVICE, dtype=torch.float32)
print(f"  device_has_fp8_mma={_device_has_fp8_mma()} (sm_{DEV_CC[0]}.{DEV_CC[1]})")
print(f"  q {tuple(q.shape)} q_s {tuple(q_s.shape)} k {tuple(k.shape)} k_s {tuple(k_s.shape)}")
score = fp8_index(q, q_s, k, k_s)
torch.cuda.synchronize()
assert_shape(score, (B, M, N), "score")
assert score.dtype == torch.float32, f"score dtype {score.dtype}"
assert torch.isfinite(score).all(), "score has nan/inf"
# Sanity: ReLU -> non-negative, scaled by q_s and k_s -> non-negative
assert (score >= 0).all(), "score should be non-negative (ReLU * positive scales)"
print("  fp8_index dispatcher OK on sm_86")

# ----- 3. NSA backend selector accepts tilelang for both prefill+decode ----
heading("ServerArgs accepts --nsa-prefill-backend tilelang --nsa-decode-backend tilelang")
from sglang.srt.server_args import ServerArgs

# Build a minimal ServerArgs to verify the override is honored. Skip
# auto-config dispatch by short-circuiting: we don't need a real model here.
sa = ServerArgs(
    model_path="deepseek-ai/DeepSeek-V4-Flash",
    nsa_prefill_backend="tilelang",
    nsa_decode_backend="tilelang",
    skip_server_warmup=True,
)
assert sa.nsa_prefill_backend == "tilelang", sa.nsa_prefill_backend
assert sa.nsa_decode_backend == "tilelang", sa.nsa_decode_backend
print(f"  nsa_prefill_backend={sa.nsa_prefill_backend} nsa_decode_backend={sa.nsa_decode_backend} OK")

# ----- 4. Auto-selector behaviour on non-Hopper bfloat16 --------------------
heading("Auto-selector picks Hopper-only backends on Ampere bfloat16 (regression target)")
sa_default = ServerArgs(
    model_path="deepseek-ai/DeepSeek-V4-Flash",
    skip_server_warmup=True,
)
sa_default.kv_cache_dtype = "bfloat16"
sa_default._set_default_nsa_backends("bfloat16", DEV_CC[0])
print(f"  auto-selected on sm_{DEV_CC[0]}.{DEV_CC[1]} bf16: prefill={sa_default.nsa_prefill_backend} decode={sa_default.nsa_decode_backend}")
# This is the upstream bug we're patching: sm<10 bf16 picks flashmla_sparse + fa3 (Hopper-only).
# Document the current behaviour; the patch we author will change this to tilelang.
if DEV_CC[0] < 10:
    if sa_default.nsa_prefill_backend == "tilelang" and sa_default.nsa_decode_backend == "tilelang":
        print("  upstream auto-selector now picks tilelang on sm<10 -- patch landed?")
    else:
        print(f"  EXPECTED-UPSTREAM-BUG: auto-selector picked Hopper-only backends on sm_{DEV_CC[0]}.{DEV_CC[1]}; --nsa-*-backend tilelang override is required")

print("\n[probe] all checks passed")
sys.exit(0)
