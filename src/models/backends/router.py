"""Backend router for selecting an ASR backend at runtime.

Typical usage:
- Short audio -> Qwen3-ASR remote
- Long audio / speaker diarization -> VibeVoice-ASR remote
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.models.backends.base import ASRBackend
from src.models.backends.remote_utils import audio_input_to_wav_bytes

logger = logging.getLogger(__name__)


class RouterBackend(ASRBackend):
    """Route `transcribe()` calls to different backends based on the request."""

    def __init__(
        self,
        *,
        short_backend: ASRBackend,
        long_backend: ASRBackend,
        long_audio_threshold_s: float = 60.0,
        force_vibevoice_when_with_speaker: bool = True,
    ):
        self.short_backend = short_backend
        self.long_backend = long_backend
        self.long_audio_threshold_s = float(long_audio_threshold_s)
        self.force_vibevoice_when_with_speaker = bool(force_vibevoice_when_with_speaker)

    def load(self) -> None:
        self.short_backend.load()
        self.long_backend.load()

    @property
    def supports_streaming(self) -> bool:
        return self.short_backend.supports_streaming or self.long_backend.supports_streaming

    @property
    def supports_hotwords(self) -> bool:
        return self.short_backend.supports_hotwords or self.long_backend.supports_hotwords

    @property
    def supports_speaker(self) -> bool:
        return self.short_backend.supports_speaker or self.long_backend.supports_speaker

    def transcribe(self, audio_input, hotwords: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # Decide routing based on duration and whether diarization is requested.
        with_speaker = bool(kwargs.get("with_speaker", False))

        _wav_bytes, duration_s = audio_input_to_wav_bytes(audio_input)

        if with_speaker and self.force_vibevoice_when_with_speaker:
            backend = self._pick_speaker_backend()
        else:
            backend = self.long_backend if duration_s >= self.long_audio_threshold_s else self.short_backend

        logger.debug(
            "RouterBackend selected %s (duration=%.2fs, with_speaker=%s)",
            backend.get_info().get("name"),
            duration_s,
            with_speaker,
        )
        return backend.transcribe(audio_input, hotwords=hotwords, **kwargs)

    def _pick_speaker_backend(self) -> ASRBackend:
        # Prefer whichever backend claims to support diarization.
        if self.long_backend.supports_speaker:
            return self.long_backend
        if self.short_backend.supports_speaker:
            return self.short_backend
        return self.long_backend

    def unload(self) -> None:
        try:
            self.short_backend.unload()
        finally:
            self.long_backend.unload()

