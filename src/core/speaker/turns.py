from __future__ import annotations

from typing import Any, Dict, List


def build_speaker_turns(
    sentences: List[Dict[str, Any]],
    *,
    gap_ms: int = 800,
    min_chars: int = 1,
) -> List[Dict[str, Any]]:
    """Build merged speaker turns from diarized sentence-level output.

    A "turn" is a consecutive run of sentences by the same speaker, optionally
    merged when the gap between sentences is small.

    Args:
        sentences: List of sentence dicts. Expected keys (best-effort):
            - speaker (str)
            - speaker_id (int)
            - start (ms int)
            - end (ms int)
            - text (str)
        gap_ms: Merge sentences when next.start - prev.end <= gap_ms.
        min_chars: Drop turns whose merged text has fewer than this many chars.

    Returns:
        List of turn dicts with keys:
            - speaker, speaker_id, start, end, text, sentence_count
    """
    if not sentences:
        return []

    if gap_ms < 0:
        gap_ms = 0
    if min_chars < 0:
        min_chars = 0

    turns: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    def _flush() -> None:
        nonlocal current
        if current is None:
            return
        text = str(current.get("text") or "")
        if len(text.strip()) >= min_chars:
            turns.append(current)
        current = None

    for sent in sentences:
        speaker = sent.get("speaker", "未知")
        speaker_id = sent.get("speaker_id", -1)
        start = sent.get("start", 0)
        end = sent.get("end", 0)
        text = sent.get("text", "")

        try:
            start_ms = int(start)
        except (TypeError, ValueError):
            start_ms = 0
        try:
            end_ms = int(end)
        except (TypeError, ValueError):
            end_ms = start_ms

        speaker_str = str(speaker) if speaker is not None else "未知"
        try:
            speaker_id_int = int(speaker_id)
        except (TypeError, ValueError):
            speaker_id_int = -1

        text_str = str(text) if text is not None else ""

        if current is None:
            current = {
                "speaker": speaker_str,
                "speaker_id": speaker_id_int,
                "start": start_ms,
                "end": end_ms,
                "text": text_str,
                "sentence_count": 1,
            }
            continue

        same_speaker = (
            current.get("speaker_id") == speaker_id_int and current.get("speaker") == speaker_str
        )
        gap = start_ms - int(current.get("end", 0) or 0)
        can_merge = same_speaker and gap <= gap_ms

        if can_merge:
            current["end"] = max(int(current.get("end", 0) or 0), end_ms)
            current["text"] = f"{current.get('text', '')}{text_str}"
            current["sentence_count"] = int(current.get("sentence_count", 0) or 0) + 1
        else:
            _flush()
            current = {
                "speaker": speaker_str,
                "speaker_id": speaker_id_int,
                "start": start_ms,
                "end": end_ms,
                "text": text_str,
                "sentence_count": 1,
            }

    _flush()
    return turns

