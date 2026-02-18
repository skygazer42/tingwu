from __future__ import annotations

from typing import Any, Dict, List

from src.core.speaker import SpeakerLabeler, build_speaker_turns
from src.core.speaker.external_diarizer_types import ExternalDiarizerSegmentLike


def segments_to_turns(
    segments: List[ExternalDiarizerSegmentLike],
    *,
    gap_ms: int = 800,
    label_style: str = "zh",
) -> List[Dict[str, Any]]:
    """Convert diarizer segments into merged speaker turns (without text).

    The returned turns include speaker labels and a stable speaker_id mapping,
    but `text` will be empty until ASR fills it later.
    """
    if not segments:
        return []

    sentences: List[Dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        sentences.append(
            {
                "spk": seg.get("spk"),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": "",
            }
        )

    labeler = SpeakerLabeler(label_style=label_style)
    labeled = labeler.label_speakers(sentences, spk_key="spk")
    return build_speaker_turns(labeled, gap_ms=gap_ms, min_chars=0)

