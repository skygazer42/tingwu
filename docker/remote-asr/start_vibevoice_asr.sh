#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${VIBEVOICE_MODEL_ID:-microsoft/VibeVoice-ASR}"
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

# Download model weights (ModelScope or HuggingFace).
if [ "${MODEL_SOURCE}" = "modelscope" ]; then
  echo "[vibevoice-asr] Downloading model from ModelScope..."
  pip install -q modelscope 2>/dev/null || true
  MODEL_PATH="$(python3 - <<PY
from modelscope import snapshot_download
print(snapshot_download('${MODEL_ID}'))
PY
)"
else
  echo "[vibevoice-asr] Downloading model from HuggingFace..."
  MODEL_PATH="$(python3 - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download('${MODEL_ID}'))
PY
)"
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
