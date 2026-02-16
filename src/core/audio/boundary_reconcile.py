"""Boundary reconciliation helpers for long-audio chunking.

This module is intentionally numpy-only and backend-agnostic so we can unit test
the boundary logic without importing the full engine (which pulls heavy deps).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

from src.core.audio.pcm import float32_to_pcm16le_bytes

__all__ = ["build_boundary_bridge_results"]


def build_boundary_bridge_results(
    audio: np.ndarray,
    chunk_results: List[Dict[str, Any]],
    *,
    sample_rate: int,
    overlap_duration_s: float,
    window_half_s: float,
    transcribe_pcm16le: Callable[[bytes], str],
) -> List[Dict[str, Any]]:
    """Create "bridge" results by re-transcribing windows around chunk boundaries.

    The returned list follows the same schema as `AudioChunker.process_parallel`:
      {"start_sample": int, "end_sample": int, "success": bool, "result": {"text": str, "sentences": []}}

    Notes:
    - We approximate the split point as `prev_end - overlap_samples` because each
      chunk includes `overlap_duration_s` audio after the split.
    - For stable merge ordering, we force each bridge's `start_sample` to sort
      *before* the next chunk (even if `window_half_s` < overlap_duration_s).
    """
    if audio is None or len(audio) == 0:
        return []
    if not chunk_results:
        return []
    if window_half_s <= 0.0:
        return []

    sr = int(sample_rate) if int(sample_rate or 0) > 0 else 16000
    window_half_samples = int(float(window_half_s) * sr)
    if window_half_samples <= 0:
        return []

    overlap_samples = int(float(overlap_duration_s) * sr)

    sorted_results = sorted(chunk_results, key=lambda x: x.get("start_sample", 0) or 0)
    bridges: List[Dict[str, Any]] = []

    for i in range(len(sorted_results) - 1):
        prev_r = sorted_results[i]
        next_r = sorted_results[i + 1]
        if not prev_r.get("success") or not next_r.get("success"):
            continue

        prev_end = int(prev_r.get("end_sample", 0) or 0)
        next_start = int(next_r.get("start_sample", 0) or 0)
        if prev_end <= 0 or next_start <= 0:
            continue

        split_sample = prev_end - overlap_samples if overlap_samples > 0 else prev_end
        split_sample = max(0, min(int(split_sample), len(audio)))

        w_start = max(0, split_sample - window_half_samples)
        w_end = min(len(audio), split_sample + window_half_samples)
        if w_end <= w_start:
            continue

        w_audio = audio[w_start:w_end].astype(np.float32, copy=False)
        pcm = float32_to_pcm16le_bytes(w_audio)
        text = str(transcribe_pcm16le(pcm) or "")
        if not text.strip():
            continue

        # Ensure the bridge is merged before the next chunk.
        sort_start = max(0, next_start - 1)

        bridges.append(
            {
                "start_sample": sort_start,
                "end_sample": int(w_end),
                "success": True,
                "result": {"text": text, "sentences": []},
                "_bridge_window": {"start_sample": int(w_start), "end_sample": int(w_end)},
            }
        )

    return bridges

