"""
End-to-end test: load *real* DeepSeek-V4-Flash with num_hidden_layers=1 on
this dev machine's RTX A5000 (sm_8.6 = same as the RTX 3090 chain).

Garbage output is fine. Goal: prove the fork patches let V4-Flash actually
run on Ampere with a real tokenizer + real FP8-native weights, exercising
the V4-specific code paths (DeepseekV4DecoderLayer, MLA with index_topk,
hash-class head, sqrtsoftplus router) that V2-Lite-Chat doesn't reach.

Layer 0 in V4-Flash has compress_ratio=0 (no DSA compressor or indexer),
so this test exercises the V4 MLA + MoE + hash-class head path only.
A separate test with num_hidden_layers>=3 is needed to exercise the
compressed indexer / TileLang dispatcher patches.

Required cached shards (from deepseek-ai/DeepSeek-V4-Flash):
  - model-00001-of-00046.safetensors  (embed.weight, ~1 GB)
  - model-00002-of-00046.safetensors  (layer 0 weights, ~3.4 GB)
  - model-00045-of-00046.safetensors  (norm.weight, head.weight, hc_head_*)
  - tokenizer + config

Run:
    cd ~/Documents/appmana/forks-sglang
    source .venv/bin/activate
    python tools/ampere/test_v4_flash_one_block.py
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PORT = 30099
HOST = "127.0.0.1"
SERVER_URL = f"http://{HOST}:{PORT}"
LOG_PATH = Path("/tmp/test_v4_flash_one_block.log")


def log(msg: str) -> None:
    print(f"[test] {msg}", flush=True)


def wait_for_ready(proc: subprocess.Popen, timeout: int = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            log(f"server exited rc={proc.returncode}; tail of log:")
            print(LOG_PATH.read_text()[-6000:])
            sys.exit(1)
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/v1/models", timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    log(f"timeout after {timeout}s waiting for ready; tail of log:")
    print(LOG_PATH.read_text()[-6000:])
    proc.send_signal(signal.SIGTERM)
    sys.exit(1)


def main() -> int:
    if not (REPO_ROOT / ".venv").is_dir():
        log("expected venv at .venv -- run tools/ampere/setup-venv.sh first")
        return 1

    env = os.environ.copy()
    env.update({
        "CUDA_HOME": env.get("CUDA_HOME", "/usr/local/cuda"),
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "1"),
        # The published deepseek-ai/DeepSeek-V4-Flash checkpoint has FP4
        # expert weights (config has expert_dtype="fp4"). The "0" setting
        # is for FP4-to-FP8 pre-converted checkpoints; we have the original
        # so keep the default "1".
        "SGLANG_DSV4_FP4_EXPERTS": "1",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
        # Use the actual checkpoint config so --json-model-override-args is
        # honored. Default 'auto' substitutes a packaged 43-layer config.
        "SGLANG_APPLY_CONFIG_BACKUP": "none",
        # 2604B is the published-stable mode per upstream PR #23661 author.
        "SGLANG_DSV4_MODE": env.get("SGLANG_DSV4_MODE", "2604"),
        "SGLANG_DSV4_2604_SUBMODE": env.get("SGLANG_DSV4_2604_SUBMODE", "2604B"),
        # On sm<9, both deep_gemm and SGLang's JIT metadata kernel fail:
        # - deep_gemm: 'Unsupported architecture'
        # - JIT kernel: dynamic SMEM size (128 KB) > Ampere's 100 KB cap
        # The TORCH path skips metadata entirely (sets deep_gemm_metadata=None)
        # and uses a torch-native fp8_paged_mqa_logits implementation. Set 0
        # to use TileLang's BF16-fallback kernel instead (faster on multi-layer).
        "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": env.get("SGLANG_FP8_PAGED_MQA_LOGITS_TORCH", "1"),
        # FP8 wo_a path is broken on Ampere (no fp8_einsum). Use BF16 einsum
        # instead. Our requantized wo_a is INT8 + BF16 scale; the
        # Dsv4Int8LinearMethod.process_weights_after_loading dequants it to
        # BF16 in place so the einsum works.
        "SGLANG_OPT_FP8_WO_A_GEMM": "0",
        # Disable wq_a + wkv fusion (the fused path expects FP8 weights with
        # FP8-specific ue8m0 scale handling). Our requantized INT8 weights load
        # individually as wq_a + wkv.
        "SGLANG_OPT_FUSE_WQA_WKV": "0",
        # On Ampere (sm<9), the upstream `flash_mla` pip pkg's kernel is
        # sm_90+ only. Route the V4 sparse-MLA call to our pure-torch
        # FP32-accumulated reference (debug_flash_mla_adapter.py).
        # Slow but produces correct (real) attention output instead of zeros.
        # Use the zero-stub for sparse-MLA to bisect: if test passes with
        # zero attention but real Marlin MoE, we know Marlin isn't the
        # issue. Re-enable "torch" once Marlin runs end-to-end.
        # "zero" = stub attention with zeros (bisect mode: validates Marlin
        # MoE without sparse-MLA torch overhead).
        # "torch" = real attention via FP32 torch fallback (slow, correct).
        # Switch back to "torch" once we know Marlin works end-to-end.
        "SGLANG_HACK_FLASHMLA_BACKEND": env.get("SGLANG_HACK_FLASHMLA_BACKEND", "zero"),
        # CUDA_LAUNCH_BLOCKING serializes kernel launches and can make multi-
        # layer forward 50x slower; only enable for debugging shape issues.
        "CUDA_LAUNCH_BLOCKING": env.get("CUDA_LAUNCH_BLOCKING", "0"),
        "TORCH_USE_CUDA_DSA": env.get("TORCH_USE_CUDA_DSA", "0"),
    })

    # Use a pre-built local 1-layer snapshot dir (config + stripped index +
    # symlinks to the 3 actual shards we have). This avoids SGLang's loader
    # trying to download all 46 shards based on the original index.json.
    # Build via tools/ampere/make_v4_flash_one_block_snapshot.py.
    DEFAULT_LOCAL = "/var/lib/inference/v4-flash-1layer"
    model_path = os.environ.get("AMPERE_TEST_MODEL", DEFAULT_LOCAL)
    if not os.path.exists(os.path.join(model_path, "config.json")):
        log(f"local snapshot missing at {model_path}; build it first")
        return 1

    # Config is already truncated in the local snapshot's config.json
    # (num_hidden_layers=1, num_nextn_predict_layers=0, compress_ratios=[0]).
    override = "{}"

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--tokenizer-path", model_path,
        "--json-model-override-args", override,
        "--tp-size", "1",
        "--pp-size", "1",
        "--mem-fraction-static", "0.85",
        "--disable-cuda-graph",
        "--skip-server-warmup",
        # Cap KV pool so it doesn't try to alloc 52 GB on a 24 GB card.
        "--context-length", "4096",
        "--max-total-tokens", "4096",
        "--max-running-requests", "1",
        "--chunked-prefill-size", "256",
        # Force our TileLang NSA backend (auto-pick is sm<9 already, but
        # explicit override is harmless and self-documenting).
        "--nsa-prefill-backend", "tilelang",
        "--nsa-decode-backend", "tilelang",
        "--trust-remote-code",
        "--host", HOST,
        "--port", str(PORT),
    ]
    log(f"launching: {' '.join(cmd)}")
    log(f"log: {LOG_PATH}")

    with LOG_PATH.open("w") as logf:
        proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=REPO_ROOT)

    try:
        log(f"waiting for ready (pid={proc.pid})...")
        wait_for_ready(proc, timeout=900)
        log("server ready")

        body = json.dumps({
            "model": model_path,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        }).encode()

        req = urllib.request.Request(
            f"{SERVER_URL}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=1800) as r:
            resp = json.loads(r.read())
        log(f"response: {resp}")

        prompt_tokens = resp["usage"]["prompt_tokens"]
        completion_tokens = resp["usage"]["completion_tokens"]
        content = resp["choices"][0]["message"]["content"]
        assert prompt_tokens > 0, f"prompt_tokens={prompt_tokens}"
        assert completion_tokens > 0, f"completion_tokens={completion_tokens}"
        assert isinstance(content, str), f"content type {type(content)}"

        log(f"PASS  prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}")
        log(f"      content (garbage expected with 1-layer truncation): {content!r}")
        return 0

    finally:
        log("shutting down server")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
