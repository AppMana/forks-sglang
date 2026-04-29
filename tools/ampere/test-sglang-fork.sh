#!/usr/bin/env bash
# 60-second local Ampere smoke test for our SGLang deepseek_v4_ampere fork.
#
# Validates the architecture path on this machine's 2x A5000 (sm_86 -- same
# compute capability as the chain's RTX 3090s) using --load-format dummy.
# Garbage output is fine; we want the forward pass to return a valid response.
#
# Usage:  tools/ampere/test-sglang-fork.sh [tp|pp]
#   tp   : --tp-size 2 --pp-size 1   (default)
#   pp   : --tp-size 1 --pp-size 2   (validates PP merge from PR #23661)

set -euo pipefail

MODE=${1:-tp}
case "$MODE" in
  tp) TP_SIZE=2; PP_SIZE=1 ;;
  pp) TP_SIZE=1; PP_SIZE=2 ;;
  *)  echo "usage: $0 [tp|pp]" >&2; exit 64 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PORT=30099
HOST=127.0.0.1
IMAGE=${SGLANG_TEST_IMAGE:-lmsysorg/sglang:dev}
LOG_FILE=$(mktemp -t sglang-test-XXXX.log)
CONTAINER_NAME=sglang-fork-test-$$

cleanup() {
  echo "[test] cleanup..."
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  rm -f "$LOG_FILE"
}
trap cleanup EXIT

echo "[test] mode=$MODE  TP=$TP_SIZE  PP=$PP_SIZE  image=$IMAGE  port=$PORT"
echo "[test] fork repo: $REPO_ROOT"

# Static import smoke (catch syntax / missing-symbol errors before launching the server).
docker run --rm --gpus all --network host \
  -v "$REPO_ROOT:/sgl-fork" \
  --entrypoint=/bin/bash \
  "$IMAGE" \
  -c "set -e; cd /sgl-fork && pip install --quiet --no-deps -e python/ 2>&1 | tail -3 && python -c 'import sglang; print(\"sglang\", sglang.__version__)' && python -c 'from sglang.srt.layers.attention.nsa.tilelang_kernel import act_quant_kernel; print(\"tilelang import OK\")'"

# Architecture smoke. Server runs detached; we wait for the ready banner, probe, tear down.
docker run -d --rm --gpus all --network host --name "$CONTAINER_NAME" \
  -v "$REPO_ROOT:/sgl-fork" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e SGLANG_DSV4_FP4_EXPERTS=0 \
  -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  --entrypoint=/bin/bash \
  "$IMAGE" \
  -c "set -e; cd /sgl-fork && pip install --quiet --no-deps -e python/ >/dev/null 2>&1 && \
      exec python -m sglang.launch_server \
        --model-path deepseek-ai/DeepSeek-V4-Flash \
        --tokenizer-path deepseek-ai/DeepSeek-V4-Flash \
        --load-format dummy \
        --tp-size $TP_SIZE --pp-size $PP_SIZE \
        --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
        --mem-fraction-static 0.7 --disable-cuda-graph \
        --trust-remote-code --host $HOST --port $PORT" \
  > "$LOG_FILE.cid" 2>&1

echo "[test] server container: $CONTAINER_NAME"

# Tail logs in background for visibility.
docker logs -f "$CONTAINER_NAME" > "$LOG_FILE" 2>&1 &
TAIL_PID=$!

# Wait up to 5 min for the ready banner.
READY=0
for i in $(seq 1 300); do
  if grep -q "The server is fired up" "$LOG_FILE" 2>/dev/null; then
    READY=1
    echo "[test] server ready after ${i}s"
    break
  fi
  if grep -qE "Traceback|Error: |fatal|Aborted" "$LOG_FILE" 2>/dev/null; then
    echo "[test] FAIL: error detected before ready banner"
    tail -50 "$LOG_FILE"
    exit 1
  fi
  if ! docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q .; then
    echo "[test] FAIL: container exited before ready banner"
    tail -50 "$LOG_FILE"
    exit 1
  fi
  sleep 1
done

if [ "$READY" -ne 1 ]; then
  echo "[test] FAIL: server did not become ready within 300s"
  tail -50 "$LOG_FILE"
  exit 1
fi

# Shape probe: garbage output is fine, we want a 200 with valid usage counts.
RESP=$(curl --silent --max-time 30 -X POST "http://$HOST:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash","messages":[{"role":"user","content":"hi"}],"max_tokens":5}')

echo "[test] response: $RESP"

# Validate shape
PROMPT_TOKENS=$(echo "$RESP" | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r["usage"]["prompt_tokens"])' 2>/dev/null || echo 0)
COMPLETION_TOKENS=$(echo "$RESP" | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r["usage"]["completion_tokens"])' 2>/dev/null || echo 0)
CONTENT=$(echo "$RESP" | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"])' 2>/dev/null || echo "")

[ "$PROMPT_TOKENS" -gt 0 ] || { echo "[test] FAIL: prompt_tokens=$PROMPT_TOKENS"; exit 1; }
[ "$COMPLETION_TOKENS" -gt 0 ] || { echo "[test] FAIL: completion_tokens=$COMPLETION_TOKENS"; exit 1; }

# Both A5000s should have non-zero memory.
GPU0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
GPU1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
echo "[test] GPU mem: gpu0=${GPU0}MiB gpu1=${GPU1}MiB"

if [ "$MODE" = "tp" ]; then
  [ "$GPU0" -gt 1024 ] && [ "$GPU1" -gt 1024 ] || { echo "[test] FAIL: TP=2 but gpu mem unbalanced"; exit 1; }
fi

echo "[test] PASS  prompt_tokens=$PROMPT_TOKENS completion_tokens=$COMPLETION_TOKENS content=\"$CONTENT\""
