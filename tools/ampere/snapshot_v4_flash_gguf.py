"""
Download a single V4-Flash GGUF quant from tecaprovn/deepseek-v4-flash-gguf.

Default: Q3_K_M (smallest quant at ~100 GB). For production use Q8_0 (~194 GB)
which keeps weights in INT8 / FP16 scales -- Ampere has INT8 tensor cores;
no FP4 emulation needed.

Run:
    cd ~/Documents/appmana/forks-sglang
    source .venv/bin/activate
    # Smallest, fastest download:
    python tools/ampere/snapshot_v4_flash_gguf.py
    # Or specify a quant:
    AMPERE_GGUF_QUANT=Q8_0 python tools/ampere/snapshot_v4_flash_gguf.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    quant = os.environ.get("AMPERE_GGUF_QUANT", "Q3_K_M").upper()
    cache_dir = os.environ.get(
        "AMPERE_GGUF_CACHE",
        "/home/administrator/.cache/huggingface/hub",
    )
    repo_id = "tecaprovn/deepseek-v4-flash-gguf"
    filename = f"DeepSeekV4-Flash-158B-{quant}.gguf"

    print(f"[snapshot] repo={repo_id} file={filename}", flush=True)
    print(f"[snapshot] cache_dir={cache_dir}", flush=True)
    path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[filename, "README.md", "*.json"],
        cache_dir=cache_dir,
        max_workers=8,
    )
    print(f"[snapshot] snapshot_dir={path}", flush=True)
    full = Path(path) / filename
    if full.exists():
        size_gb = full.stat().st_size / 1e9
        print(f"[snapshot] {full} -> {size_gb:.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
