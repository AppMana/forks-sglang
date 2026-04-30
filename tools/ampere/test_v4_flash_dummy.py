"""
Phase A end-to-end verification: run V4-Flash with --load-format dummy
(random weights, no checkpoint) so the FP4 expert weight-loading wall doesn't
fire. With Phase A's torch sparse-MLA fallback, this test exercises the full
V4 forward path on sm_86 (RTX A5000) and verifies a non-empty completion
returns.

Output is GARBAGE (random init weights → noise tokens) but that's the test
target: the *path* runs end-to-end, not the model quality.

Run:
    cd ~/Documents/appmana/forks-sglang
    source .venv/bin/activate
    python tools/ampere/test_v4_flash_dummy.py
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
LOG_PATH = Path("/tmp/test_v4_flash_dummy.log")


def log(msg: str) -> None:
    print(f"[test] {msg}", flush=True)


def wait_for_ready(proc: subprocess.Popen, timeout: int = 600) -> None:
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
        # With dummy weights, FP4=0 means experts are declared FP8 (which
        # passes through the standard fp8 quant_method.apply path on
        # Ampere with weight-only Marlin emulation) and avoids the FP4
        # half-packing assertion. Real FP4 weights need Phase D's GGUF Q8.
        "SGLANG_DSV4_FP4_EXPERTS": "0",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
        "SGLANG_APPLY_CONFIG_BACKUP": "none",
        "SGLANG_DSV4_MODE": "2604",
        "SGLANG_DSV4_2604_SUBMODE": "2604B",
        "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
        # Phase A: torch sparse-MLA fallback
        "SGLANG_HACK_FLASHMLA_BACKEND": env.get("SGLANG_HACK_FLASHMLA_BACKEND", "torch"),
    })

    DEFAULT_LOCAL = "/var/lib/inference/v4-flash-1layer"
    model_path = os.environ.get("AMPERE_TEST_MODEL", DEFAULT_LOCAL)
    if not os.path.exists(os.path.join(model_path, "config.json")):
        log(f"local snapshot missing at {model_path}; build it first")
        return 1

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--tokenizer-path", model_path,
        "--json-model-override-args", "{}",
        # Random init -- bypasses the FP4 weight-loader path entirely
        "--load-format", "dummy",
        "--tp-size", "1",
        "--pp-size", "1",
        "--mem-fraction-static", "0.7",
        "--disable-cuda-graph",
        "--context-length", "4096",
        "--max-total-tokens", "8192",
        "--max-running-requests", "4",
        "--chunked-prefill-size", "512",
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
        wait_for_ready(proc, timeout=600)
        log("server ready")

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
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read())
        log(f"response: {resp}")

        prompt_tokens = resp["usage"]["prompt_tokens"]
        completion_tokens = resp["usage"]["completion_tokens"]
        content = resp["choices"][0]["message"]["content"]
        assert prompt_tokens > 0, f"prompt_tokens={prompt_tokens}"
        assert completion_tokens > 0, f"completion_tokens={completion_tokens}"
        assert isinstance(content, str), f"content type {type(content)}"
        log(f"PASS  prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}")
        log(f"      content (random init -> garbage): {content!r}")
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
