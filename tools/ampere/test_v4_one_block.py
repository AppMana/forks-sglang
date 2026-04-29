"""
End-to-end test: load DeepSeek-V4-Flash with num_hidden_layers=1, real weights,
forward pass on this dev machine's RTX A5000 (sm_8.6 = same as RTX 3090 chain).

Garbage output is fine. Goal: prove the fork patches let V4-Flash actually
run on Ampere with a real tokenizer + real (FP8 native) attention/MoE
weights for one decoder block, returning a 200 OK from the OpenAI endpoint
with non-empty tokens.

Truncates the model via SGLang's `--json-model-override-args` so we exercise
exactly one DeepseekV4DecoderLayer instead of all 43. One layer fits in 24
GB (~6 GB attention + ~6-8 GB experts at FP8) where the full model would
need ~280 GB.

Run:
    cd ~/Documents/appmana/forks-sglang
    source .venv/bin/activate
    python tools/ampere/test_v4_one_block.py
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
LOG_PATH = Path("/tmp/test_v4_one_block.log")


def log(msg: str) -> None:
    print(f"[test] {msg}", flush=True)


def wait_for_ready(proc: subprocess.Popen, timeout: int = 600) -> None:
    """Poll the log + process state. Server is "ready" when /v1/models returns 200."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            log(f"server exited rc={proc.returncode}; tail of log:")
            print(LOG_PATH.read_text()[-4000:])
            sys.exit(1)
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/v1/models", timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    log(f"timeout after {timeout}s waiting for ready; tail of log:")
    print(LOG_PATH.read_text()[-4000:])
    proc.send_signal(signal.SIGTERM)
    sys.exit(1)


def main() -> int:
    if not (REPO_ROOT / ".venv").is_dir():
        log("expected venv at .venv -- run tools/ampere/setup-venv.sh first")
        return 1

    env = os.environ.copy()
    env.update({
        "CUDA_HOME": env.get("CUDA_HOME", "/usr/local/cuda"),
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "1"),  # GPU 1, gpu0 has gnome-rdp using ~5 GB
        "SGLANG_DSV4_FP4_EXPERTS": "0",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
        # Use the actual checkpoint config so --json-model-override-args is
        # honored. Default 'auto' substitutes a packaged 43-layer config.
        "SGLANG_APPLY_CONFIG_BACKUP": "none",
    })

    # num_hidden_layers=1 truncates the model to a single DeepseekV4DecoderLayer.
    # Embeddings + 1 layer + lm_head is small enough to fit on one 24 GB Ampere
    # card with real FP8-native weights from deepseek-ai/DeepSeek-V4-Flash.
    override = json.dumps({"num_hidden_layers": 1})

    # Use the smallest publicly available DeepseekV2 family model so we exercise
    # the same model class hierarchy (DeepseekV2DecoderLayer -> MLA -> MoE) with
    # real weights that fit on a single 24 GB Ampere card. V4-Flash itself
    # would require ~280 GB even pre-truncation just to download. V2-Lite is
    # ~32 GB total; with num_hidden_layers=1 we only need ~1 GB of layer
    # weights resident plus embeddings (~1 GB).
    model_path = os.environ.get("AMPERE_TEST_MODEL", "deepseek-ai/DeepSeek-V2-Lite-Chat")

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--tokenizer-path", model_path,
        "--json-model-override-args", override,
        "--tp-size", "1",
        "--pp-size", "1",
        "--mem-fraction-static", "0.7",
        "--disable-cuda-graph",
        # Smoke test only: caps the KV cache + max prefill so MLATokenToKVPool
        # doesn't try to allocate 52 GB on a 24 GB card.
        "--context-length", "4096",
        "--max-total-tokens", "8192",
        "--max-running-requests", "4",
        "--chunked-prefill-size", "512",
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
        wait_for_ready(proc, timeout=600)
        log("server ready")

        # Smoke completion: garbage output is fine, just need a valid response.
        body = json.dumps({
            "model": model_path,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        }).encode()

        req = urllib.request.Request(
            f"{SERVER_URL}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
        log(f"response: {resp}")

        # Validate shape, not content quality.
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
