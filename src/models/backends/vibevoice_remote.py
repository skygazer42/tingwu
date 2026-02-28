"""Remote VibeVoice-ASR backend (vLLM OpenAI-compatible chat completions API).

VibeVoice's vLLM plugin exposes `/v1/chat/completions` with `audio_url` inputs and
returns a JSON-formatted transcription with keys like:
  - Start time
  - End time
  - Speaker ID
  - Content
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from src.models.backends.base import ASRBackend
from src.models.backends.remote_utils import audio_input_to_wav_bytes

logger = logging.getLogger(__name__)


class VibeVoiceRemoteBackend(ASRBackend):
    """Call a remote VibeVoice-ASR server.

    The official deployment uses `/v1/chat/completions` (OpenAI-compatible).
    Some alternative deployments may implement `/v1/audio/transcriptions`; we
    keep a switch for that for flexibility.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_s: float = 600.0,
        use_chat_completions_fallback: bool = True,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.timeout_s = float(timeout_s)
        self.use_chat_completions_fallback = bool(use_chat_completions_fallback)
        self._client: Optional[httpx.Client] = None

    def load(self) -> None:
        if self._client is None:
            # Do not inherit HTTP(S)_PROXY/ALL_PROXY by default.
            # These backends usually talk to localhost/docker networks, and some
            # environments set unsupported proxy schemes (e.g. "socks://") that
            # cause httpx client init to fail before sending any request.
            self._client = httpx.Client(timeout=self.timeout_s, trust_env=False)

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_hotwords(self) -> bool:
        return True

    @property
    def supports_speaker(self) -> bool:
        # VibeVoice-ASR jointly performs diarization.
        return True

    def transcribe(self, audio_input, hotwords: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # Prefer chat completions for the official vLLM plugin deployment.
        if self.use_chat_completions_fallback:
            return self._transcribe_chat_completions(audio_input, hotwords=hotwords)
        return self._transcribe_audio_transcriptions(audio_input, hotwords=hotwords)

    def _transcribe_chat_completions(self, audio_input, hotwords: Optional[str]) -> Dict[str, Any]:
        self.load()
        assert self._client is not None

        audio_bytes, duration_s = audio_input_to_wav_bytes(audio_input)

        mime = "audio/wav"
        if isinstance(audio_input, (str, Path)):
            mime = _guess_mime_type(audio_input)

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        data_url = f"data:{mime};base64,{audio_b64}"

        show_keys = ["Start time", "End time", "Speaker ID", "Content"]
        if duration_s > 0:
            if hotwords:
                prompt_text = (
                    f"This is a {duration_s:.2f} seconds audio, with extra info: {hotwords.strip()}\n\n"
                    f"Please transcribe it with these keys: {', '.join(show_keys)}"
                )
            else:
                prompt_text = (
                    f"This is a {duration_s:.2f} seconds audio, please transcribe it with these keys: "
                    + ", ".join(show_keys)
                )
        else:
            if hotwords:
                prompt_text = (
                    f"Please transcribe this audio with extra info: {hotwords.strip()}\n\n"
                    f"Return JSON with keys: {', '.join(show_keys)}"
                )
            else:
                prompt_text = f"Please transcribe this audio and return JSON with keys: {', '.join(show_keys)}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that transcribes audio input into text output in JSON format.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": data_url}},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ],
            "max_tokens": 32768,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "stream": False,
        }

        url = f"{self.base_url}/v1/chat/completions"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()

        raw = resp.json()
        content = _extract_chat_content(raw)
        segments = _parse_vibevoice_segments(content)

        if not segments:
            return {"text": content.strip(), "sentence_info": []}

        sentence_info: List[Dict[str, Any]] = []
        texts: List[str] = []
        for seg in segments:
            text = str(seg.get("text") or "").strip()
            if not text:
                continue
            start_ms = _time_to_ms(seg.get("start_time"))
            end_ms = _time_to_ms(seg.get("end_time"))
            spk = seg.get("speaker_id")
            sentence_info.append({"text": text, "start": start_ms, "end": end_ms, "spk": spk})
            texts.append(text)

        return {"text": " ".join(texts).strip(), "sentence_info": sentence_info}

    def _transcribe_audio_transcriptions(self, audio_input, hotwords: Optional[str]) -> Dict[str, Any]:
        """Optional OpenAI transcription endpoint support (if a server provides it)."""
        self.load()
        assert self._client is not None

        audio_bytes, _duration_s = audio_input_to_wav_bytes(audio_input)
        url = f"{self.base_url}/v1/audio/transcriptions"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {"model": self.model}
        if hotwords:
            data["prompt"] = hotwords
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}

        resp = self._client.post(url, data=data, files=files, headers=headers)
        resp.raise_for_status()
        obj = resp.json()
        if isinstance(obj, dict):
            text = str(obj.get("text") or obj.get("transcript") or "")
        else:
            text = str(obj)
        return {"text": text, "sentence_info": []}


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
        return "video/mp4"
    return "application/octet-stream"


def _extract_chat_content(obj: object) -> str:
    if not isinstance(obj, dict):
        return str(obj)
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0] or {}
        if isinstance(c0, dict):
            msg = c0.get("message") or {}
            if isinstance(msg, dict):
                content = msg.get("content")
                if content is not None:
                    return str(content)
    return ""


def _parse_vibevoice_segments(text: str) -> List[Dict[str, Any]]:
    """Parse the model JSON output into a normalized list of segments."""
    if not text:
        return []

    json_str = _extract_json_str(text)
    if not json_str:
        return []

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return []

    if isinstance(obj, dict):
        items = [obj]
    elif isinstance(obj, list):
        items = obj
    else:
        return []

    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        out.append(
            {
                "start_time": item.get("start_time", item.get("Start time", item.get("Start"))),
                "end_time": item.get("end_time", item.get("End time", item.get("End"))),
                "speaker_id": item.get("speaker_id", item.get("Speaker ID", item.get("Speaker"))),
                "text": item.get("text", item.get("Content", item.get("content"))),
            }
        )

    # Drop empty items
    return [x for x in out if isinstance(x.get("text"), (str, int, float)) and str(x.get("text")).strip()]


def _extract_json_str(text: str) -> str:
    """Best-effort extraction of JSON from the model output."""
    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()

    # Find the first JSON-looking bracket.
    lb = text.find("[")
    lc = text.find("{")
    starts = [x for x in (lb, lc) if x != -1]
    if not starts:
        return ""
    start = min(starts)

    # Naive bracket matching (good enough for typical model output).
    depth = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end != -1:
        return text[start:end]
    return text[start:].strip()


def _time_to_ms(v: object) -> int:
    if v is None:
        return 0
    if isinstance(v, (int, float)):
        return int(round(float(v) * 1000.0))

    s = str(v).strip()
    if not s:
        return 0
    low = s.lower()
    if low.endswith("s"):
        low = low[:-1].strip()
    if ":" in low:
        parts = low.split(":")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return 0
        if len(nums) == 2:
            minutes, seconds = nums
            return int(round((minutes * 60.0 + seconds) * 1000.0))
        if len(nums) == 3:
            hours, minutes, seconds = nums
            return int(round((hours * 3600.0 + minutes * 60.0 + seconds) * 1000.0))
        return 0
    try:
        return int(round(float(low) * 1000.0))
    except ValueError:
        return 0
