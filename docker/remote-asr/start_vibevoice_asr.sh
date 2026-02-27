#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${VIBEVOICE_MODEL_ID:-microsoft/VibeVoice-ASR}"
MODEL_PATH_OVERRIDE="${VIBEVOICE_MODEL_PATH:-}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-vibevoice}"
DTYPE="${DTYPE:-float16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"
FFMPEG_MAX_CONCURRENCY="${VIBEVOICE_FFMPEG_MAX_CONCURRENCY:-16}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"

echo "[vibevoice-asr] MODEL_ID=${MODEL_ID}"
if [ -n "${MODEL_PATH_OVERRIDE}" ]; then
  echo "[vibevoice-asr] VIBEVOICE_MODEL_PATH=${MODEL_PATH_OVERRIDE}"
fi
echo "[vibevoice-asr] PORT=${PORT}"
echo "[vibevoice-asr] SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "[vibevoice-asr] DTYPE=${DTYPE}"
echo "[vibevoice-asr] MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "[vibevoice-asr] MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "[vibevoice-asr] MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "[vibevoice-asr] GPU_MEMORY_UTILIZATION=${GPU_MEM_UTIL}"
echo "[vibevoice-asr] VIBEVOICE_FFMPEG_MAX_CONCURRENCY=${FFMPEG_MAX_CONCURRENCY}"
echo "[vibevoice-asr] MODEL_SOURCE=${MODEL_SOURCE}"

export VIBEVOICE_FFMPEG_MAX_CONCURRENCY="${FFMPEG_MAX_CONCURRENCY}"

if [ ! -f "/app/pyproject.toml" ] && [ ! -f "/app/setup.py" ]; then
  echo "[vibevoice-asr] ERROR: /app does not look like a VibeVoice repo." >&2
  echo "[vibevoice-asr] Please set VIBEVOICE_REPO_PATH to a directory containing vllm_plugin + python package." >&2
  echo "[vibevoice-asr] Current /app contents:" >&2
  ls -la /app >&2 || true
  exit 2
fi

apt-get -o Acquire::Retries=3 update || {
  echo "[apt] update failed; falling back to http://mirrors.aliyun.com/ubuntu" >&2
  if [ -f /etc/apt/sources.list ]; then
    sed -i \
      's|https://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g; s|https://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g; s|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g; s|https://mirrors.aliyun.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
      /etc/apt/sources.list || true
  fi
  apt-get -o Acquire::Retries=3 update
}

apt-get install -y ffmpeg libsndfile1

# Install VibeVoice from mounted repo with vLLM support.
python3 -m pip install -e "/app[vllm]"

_ensure_modelscope() {
  if python3 -c "import modelscope" >/dev/null 2>&1; then
    return 0
  fi
  echo "[vibevoice-asr] ModelScope python package not found; installing..." >&2
  python3 -m pip install -q modelscope || true
  if python3 -c "import modelscope" >/dev/null 2>&1; then
    return 0
  fi
  echo "[vibevoice-asr] WARNING: failed to import modelscope after install" >&2
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
    print(f"[vibevoice-asr] ModelScope snapshot_download failed: {e}", file=sys.stderr)
PY
}

_download_huggingface() {
  python3 - <<PY
import sys
try:
    from huggingface_hub import snapshot_download
    path = snapshot_download("${MODEL_ID}")
    if path:
        print(path)
except Exception as e:
    print(f"[vibevoice-asr] HuggingFace snapshot_download failed: {e}", file=sys.stderr)
PY
}

# Download model weights (ModelScope / HuggingFace / local override).
MODEL_PATH=""
if [ -n "${MODEL_PATH_OVERRIDE}" ]; then
  MODEL_PATH="${MODEL_PATH_OVERRIDE}"
elif [ "${MODEL_SOURCE}" = "modelscope" ]; then
  echo "[vibevoice-asr] Downloading model from ModelScope..."
  if _ensure_modelscope; then
    # ModelScope may print progress/info lines to stdout; keep only the last non-empty line
    # as the actual snapshot path.
    MODEL_PATH="$(_download_modelscope | awk 'NF{line=$0} END{print line}' || true)"
  fi
fi

if [ -z "${MODEL_PATH}" ]; then
  echo "[vibevoice-asr] Falling back to HuggingFace download..." >&2
  MODEL_PATH="$(_download_huggingface | awk 'NF{line=$0} END{print line}' || true)"
fi

if [ -z "${MODEL_PATH}" ]; then
  echo "[vibevoice-asr] ERROR: failed to resolve model path." >&2
  echo "[vibevoice-asr] - If you are offline: mount weights and set VIBEVOICE_MODEL_PATH=/path" >&2
  echo "[vibevoice-asr] - Or set MODEL_SOURCE=modelscope and ensure modelscope is installable" >&2
  exit 2
fi

echo "[vibevoice-asr] MODEL_PATH=${MODEL_PATH}"

# Tokenizer files are required for vLLM to serve this repo. Keep best-effort.
python3 -m vllm_plugin.tools.generate_tokenizer_files --output "${MODEL_PATH}" || true

# NOTE: official docs use 64k context, but that usually needs >=24GB VRAM.
# For smoke tests on ~16GB VRAM GPUs, use smaller limits by default.
exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --dtype "${DTYPE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --enforce-eager \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --chat-template-content-format openai \
  --tensor-parallel-size 1 \
  --allowed-local-media-path /app \
  --port "${PORT}"
