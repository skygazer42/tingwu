#!/usr/bin/env bash
set -uo pipefail

# Smoke-test TingWu HTTP endpoints across multiple ports (multi-model deployment).
#
# Usage:
#   scripts/smoke_all_endpoints.sh
#
# Options (env):
#   PORTS="8101 8102 ..."   Ports to test (default: common TingWu ports)
#   AUDIO="data/benchmark/test_short.mp3"  Audio file for /transcribe tests
#   TIMEOUT_S=10            Per-request curl timeout seconds
#   DIARIZER_PORT=8300      Optional diarizer port (set empty to skip)
#
# Notes:
# - This is a best-effort smoke test. Some backends require extra model artifacts
#   (e.g. GGUF) or remote services (Qwen3/VibeVoice wrappers).

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PORTS="${PORTS:-8101 8102 8103 8104 8105 8201 8202}"
DIARIZER_PORT="${DIARIZER_PORT:-8300}"
TIMEOUT_S="${TIMEOUT_S:-10}"
AUDIO="${AUDIO:-data/benchmark/test_short.mp3}"

if [ ! -f "${AUDIO}" ]; then
  echo "ERROR: AUDIO not found: ${AUDIO}" >&2
  exit 2
fi

PASS=0
FAIL=0

_ok() {
  PASS=$((PASS + 1))
  echo "OK   $*"
}

_fail() {
  FAIL=$((FAIL + 1))
  echo "FAIL $*" >&2
}

_curl_json() {
  # Usage: _curl_json URL [curl args...]
  local url="$1"; shift || true
  curl -fsS -m "${TIMEOUT_S}" "${url}" "$@" | python3 -m json.tool >/dev/null
}

_curl_text() {
  local url="$1"; shift || true
  curl -fsS -m "${TIMEOUT_S}" "${url}" "$@" >/dev/null
}

_test_one_base() {
  local base="$1"

  echo ""
  echo "=============================="
  echo "== BASE=${base}"
  echo "=============================="

  if _curl_json "${base}/health"; then _ok "${base} GET /health"; else _fail "${base} GET /health"; fi
  if _curl_json "${base}/openapi.json"; then _ok "${base} GET /openapi.json"; else _fail "${base} GET /openapi.json"; fi
  if _curl_json "${base}/api/v1/backend"; then _ok "${base} GET /api/v1/backend"; else _fail "${base} GET /api/v1/backend"; fi

  if _curl_json "${base}/metrics"; then _ok "${base} GET /metrics"; else _fail "${base} GET /metrics"; fi
  if _curl_text "${base}/metrics/prometheus"; then _ok "${base} GET /metrics/prometheus"; else _fail "${base} GET /metrics/prometheus"; fi

  if _curl_json "${base}/config"; then _ok "${base} GET /config"; else _fail "${base} GET /config"; fi
  if _curl_json "${base}/config/all"; then _ok "${base} GET /config/all"; else _fail "${base} GET /config/all"; fi
  if _curl_json "${base}/config/reload" -X POST; then _ok "${base} POST /config/reload"; else _fail "${base} POST /config/reload"; fi

  if _curl_json "${base}/api/v1/hotwords"; then _ok "${base} GET /api/v1/hotwords"; else _fail "${base} GET /api/v1/hotwords"; fi
  if _curl_json "${base}/api/v1/hotwords/context"; then _ok "${base} GET /api/v1/hotwords/context"; else _fail "${base} GET /api/v1/hotwords/context"; fi
  if _curl_json "${base}/api/v1/hotwords/reload" -X POST; then _ok "${base} POST /api/v1/hotwords/reload"; else _fail "${base} POST /api/v1/hotwords/reload"; fi
  if _curl_json "${base}/api/v1/hotwords/context/reload" -X POST; then _ok "${base} POST /api/v1/hotwords/context/reload"; else _fail "${base} POST /api/v1/hotwords/context/reload"; fi

  if curl -fsS -m "${TIMEOUT_S}" -X POST "${base}/api/v1/transcribe" \
      -F "file=@${AUDIO}" \
      -F "with_speaker=false" \
      -F "apply_hotword=true" \
      -F "apply_llm=false" \
    | python3 -m json.tool >/dev/null; then
    _ok "${base} POST /api/v1/transcribe"
  else
    _fail "${base} POST /api/v1/transcribe"
  fi

  if curl -fsS -m "${TIMEOUT_S}" -X POST "${base}/api/v1/transcribe/batch" \
      -F "files=@${AUDIO}" \
      -F "files=@${AUDIO}" \
      -F "with_speaker=false" \
      -F "apply_hotword=true" \
      -F "apply_llm=false" \
    | python3 -m json.tool >/dev/null; then
    _ok "${base} POST /api/v1/transcribe/batch"
  else
    _fail "${base} POST /api/v1/transcribe/batch"
  fi
}

_test_diarizer() {
  local base="http://localhost:${DIARIZER_PORT}"
  echo ""
  echo "=============================="
  echo "== DIARIZER=${base}"
  echo "=============================="

  if _curl_json "${base}/health"; then _ok "${base} GET /health"; else _fail "${base} GET /health"; fi
  if _curl_json "${base}/openapi.json"; then _ok "${base} GET /openapi.json"; else _fail "${base} GET /openapi.json"; fi

  # Diarizer only accepts WAV container input.
  local wav_tmp
  wav_tmp="$(mktemp -t tingwu_diarizer_XXXXXX.wav)"
  if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -nostdin -y -loglevel error -i "${AUDIO}" -ac 1 -ar 16000 -c:a pcm_s16le "${wav_tmp}" || true
  fi

  if [ ! -s "${wav_tmp}" ]; then
    echo "SKIP ${base} POST /api/v1/diarize (ffmpeg not available to generate wav)" >&2
    rm -f "${wav_tmp}" || true
    return 0
  fi

  if curl -fsS -m "${TIMEOUT_S}" -X POST "${base}/api/v1/diarize" \
      -F "file=@${wav_tmp}" \
    | python3 -m json.tool >/dev/null; then
    _ok "${base} POST /api/v1/diarize"
  else
    _fail "${base} POST /api/v1/diarize"
  fi

  rm -f "${wav_tmp}" || true
}

echo "TingWu smoke test"
echo "- PORTS=${PORTS}"
echo "- AUDIO=${AUDIO}"
echo "- TIMEOUT_S=${TIMEOUT_S}"
echo ""

for p in ${PORTS}; do
  _test_one_base "http://localhost:${p}"
done

if [ -n "${DIARIZER_PORT}" ]; then
  _test_diarizer
fi

echo ""
echo "=============================="
echo "Summary"
echo "=============================="
echo "PASS=${PASS} FAIL=${FAIL}"

if [ "${FAIL}" -gt 0 ]; then
  exit 1
fi
