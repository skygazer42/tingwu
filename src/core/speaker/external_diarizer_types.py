from __future__ import annotations

from typing import Any, Mapping, TypedDict


class ExternalDiarizerSegment(TypedDict):
    """Normalized diarization segment (milliseconds)."""

    spk: int
    start: int
    end: int


ExternalDiarizerSegmentLike = Mapping[str, Any]

