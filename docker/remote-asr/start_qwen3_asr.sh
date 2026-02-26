#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${QWEN3_MODEL_ID:-Qwen/Qwen3-ASR-1.7B}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.85}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"

echo "[qwen3-asr] MODEL_ID=${MODEL_ID}"
echo "[qwen3-asr] PORT=${PORT}"
echo "[qwen3-asr] GPU_MEMORY_UTILIZATION=${GPU_MEM_UTIL}"
echo "[qwen3-asr] MODEL_SOURCE=${MODEL_SOURCE}"

if [ "${MODEL_SOURCE}" = "modelscope" ]; then
  echo "[qwen3-asr] Downloading model from ModelScope..."
  pip install -q modelscope 2>/dev/null || true
  MODEL_PATH="$(python3 - <<PY
from modelscope import snapshot_download
print(snapshot_download('${MODEL_ID}'))
PY
)"
  echo "[qwen3-asr] MODEL_PATH=${MODEL_PATH}"
else
  MODEL_PATH="${MODEL_ID}"
fi

exec qwen-asr-serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --trust-remote-code
