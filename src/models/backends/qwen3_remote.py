"""Remote Qwen3-ASR backend (vLLM OpenAI-compatible chat completions API)."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from src.models.backends.base import ASRBackend
from src.models.backends.remote_utils import audio_input_to_wav_bytes

logger = logging.getLogger(__name__)


class Qwen3RemoteBackend(ASRBackend):
    """Call a remote Qwen3-ASR server via `/v1/chat/completions`."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 60.0,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.timeout_s = float(timeout_s)
        self._client: Optional[httpx.Client] = None

    def load(self) -> None:
        # Lazy init so tests can patch httpx.Client.post without needing a real server.
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_speaker(self) -> bool:
        return False

    def transcribe(self, audio_input, hotwords: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        self.load()
        assert self._client is not None

        wav_bytes, duration_s = audio_input_to_wav_bytes(audio_input)

        url = f"{self.base_url}/v1/chat/completions"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        mime = "audio/wav"
        if isinstance(audio_input, (str, Path)):
            mime = _guess_mime_type(audio_input)

        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        data_url = f"data:{mime};base64,{audio_b64}"

        # Qwen3-ASR vLLM server is OpenAI-compatible and supports audio_url in chat completions.
        # The model typically returns metadata + "<asr_text>..."; we extract the text part.
        user_content = [
            {"type": "audio_url", "audio_url": {"url": data_url}},
        ]
        if hotwords:
            user_content.append({"type": "text", "text": hotwords})
        elif duration_s > 0:
            # A tiny hint can improve model behavior for some deployments.
            user_content.append({"type": "text", "text": f"Audio duration: {duration_s:.2f}s"})

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0.0,
            "stream": False,
        }

        resp = self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()

        payload = resp.json()
        text = _extract_text_from_chat_completion(payload)

        return {
            "text": text,
            "sentence_info": [],
        }


def _guess_mime_type(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".wav":
        return "audio/wav"
    if ext == ".mp3":
        return "audio/mpeg"
    if ext == ".flac":
        return "audio/flac"
    if ext in (".ogg", ".opus"):
        return "audio/ogg"
    if ext in (".m4a", ".mp4", ".m4v", ".mov", ".webm"):
        # vLLM can decode audio from videos too (ffmpeg), but the exact container may vary.
        return "video/mp4"
    return "application/octet-stream"


def _extract_text_from_chat_completion(obj: object) -> str:
    """Extract transcription text from an OpenAI-style chat completion response."""
    if not isinstance(obj, dict):
        return str(obj).strip()

    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        choice0 = choices[0] or {}
        if isinstance(choice0, dict):
            msg = choice0.get("message") or {}
            if isinstance(msg, dict):
                content = msg.get("content")
                if content is not None:
                    s = str(content).strip()
                    # Qwen3-ASR commonly wraps the transcript with "<asr_text>".
                    tag = "<asr_text>"
                    if tag in s:
                        return s.split(tag, 1)[1].strip()
                    return s

    # Fallback: try common alternative keys.
    for k in ("text", "transcript", "content"):
        if k in obj:
            return str(obj.get(k) or "").strip()

    return ""
