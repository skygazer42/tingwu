#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${QWEN3_MODEL_ID:-Qwen/Qwen3-ASR-1.7B}"
MODEL_PATH_OVERRIDE="${QWEN3_MODEL_PATH:-}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.85}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"

echo "[qwen3-asr] MODEL_ID=${MODEL_ID}"
if [ -n "${MODEL_PATH_OVERRIDE}" ]; then
  echo "[qwen3-asr] QWEN3_MODEL_PATH=${MODEL_PATH_OVERRIDE}"
fi
echo "[qwen3-asr] PORT=${PORT}"
echo "[qwen3-asr] GPU_MEMORY_UTILIZATION=${GPU_MEM_UTIL}"
echo "[qwen3-asr] MODEL_SOURCE=${MODEL_SOURCE}"

_ensure_modelscope() {
  if python3 -c "import modelscope" >/dev/null 2>&1; then
    return 0
  fi

  echo "[qwen3-asr] ModelScope python package not found; installing..." >&2
  python3 -m pip install -q modelscope || true

  if python3 -c "import modelscope" >/dev/null 2>&1; then
    return 0
  fi

  echo "[qwen3-asr] WARNING: failed to import modelscope after install; falling back to HF download" >&2
  return 1
}

_download_modelscope() {
  python3 - <<PY
import sys
try:
    from modelscope import snapshot_download
    path = snapshot_download("${MODEL_ID}")
    if path:
        print(path)
except Exception as e:
    print(f"[qwen3-asr] ModelScope snapshot_download failed: {e}", file=sys.stderr)
PY
}

MODEL_PATH=""
if [ -n "${MODEL_PATH_OVERRIDE}" ]; then
  MODEL_PATH="${MODEL_PATH_OVERRIDE}"
elif [ "${MODEL_SOURCE}" = "modelscope" ]; then
  echo "[qwen3-asr] Downloading model from ModelScope..."
  if _ensure_modelscope; then
    # ModelScope may print progress/info lines to stdout; keep only the last non-empty line
    # as the actual snapshot path.
    MODEL_PATH="$(_download_modelscope | awk 'NF{line=$0} END{print line}' || true)"
  fi
fi

if [ -n "${MODEL_PATH}" ]; then
  echo "[qwen3-asr] MODEL_PATH=${MODEL_PATH}"
else
  echo "[qwen3-asr] Using model id for download: ${MODEL_ID}" >&2
  MODEL_PATH="${MODEL_ID}"
fi

exec qwen-asr-serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --trust-remote-code
