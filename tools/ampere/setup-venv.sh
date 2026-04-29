#!/usr/bin/env bash
# One-time venv setup for local Ampere development on this machine
# (CUDA 13 toolkit at /usr/local/cuda, 2x A5000 sm_86).
# Idempotent — re-runnable to fix stale state.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"

# Some files in python/sglang/srt/grpc are written as root by docker runs that
# bind-mounted this dir. Reset ownership so the venv build can write.
if find python -not -user "$USER" 2>/dev/null | head -1 | grep -q .; then
  echo "[setup] fixing ownership of root-owned files in python/"
  sudo chown -R "$USER:$USER" python/
fi

if [ ! -d .venv ]; then
  uv venv .venv --python=3.12
fi

# shellcheck disable=SC1091
source .venv/bin/activate
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

uv pip install --upgrade pip uv >/dev/null
uv pip install "torch==2.9.1" "torchvision" --extra-index-url https://download.pytorch.org/whl/cu130
uv pip install -e ./python
# Override the pip-resolved sgl-kernel with the cu130 prebuilt wheel.
SGL_KERNEL_VERSION=$(uv pip show sgl-kernel | awk '/^Version:/ {print $2}' | sed 's/+.*//')
uv pip install --force-reinstall \
  "https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_x86_64.whl"
uv pip install tilelang

echo
echo "[setup] DONE. activate with: source $REPO_ROOT/.venv/bin/activate"
echo "[setup] then run: tools/ampere/test-sglang-fork.sh tp"
