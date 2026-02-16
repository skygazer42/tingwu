import numpy as np

from src.core.audio.boundary_reconcile import build_boundary_bridge_results
from src.core.audio.chunker import AudioChunker


def test_boundary_bridge_injection_improves_text_merge():
    sr = 16000
    audio = np.zeros((int(sr * 2.2),), dtype=np.float32)

    # Two overlapped chunks (e.g. time-based chunking):
    # - chunk1: [0.0s, 2.0s]
    # - chunk2: [1.0s, 2.2s]
    chunk_results = [
        {
            "start_sample": 0,
            "end_sample": int(sr * 2.0),
            "success": True,
            "result": {"text": "hello", "sentences": []},
        },
        {
            "start_sample": int(sr * 1.0),
            "end_sample": len(audio),
            "success": True,
            "result": {"text": "orld", "sentences": []},
        },
    ]

    chunker = AudioChunker()
    out_no_bridge = chunker.merge_results(chunk_results, sample_rate=sr, overlap_chars=20)["text"]
    assert out_no_bridge == "helloorld"

    # Build a boundary window that should transcribe the missing "w".
    # (window_half_s can be smaller than overlap_duration_s; the bridge ordering
    # is handled by the helper to ensure it merges before the next chunk.)
    def transcribe_pcm16le(pcm: bytes) -> str:
        if len(pcm) == 25600:  # 0.8s @ 16kHz mono PCM16LE
            return "world"
        return ""

    bridges = build_boundary_bridge_results(
        audio,
        chunk_results,
        sample_rate=sr,
        overlap_duration_s=0.5,
        window_half_s=0.4,
        transcribe_pcm16le=transcribe_pcm16le,
    )
    assert len(bridges) == 1

    out_with_bridge = chunker.merge_results(chunk_results + bridges, sample_rate=sr, overlap_chars=20)["text"]
    assert out_with_bridge == "helloworld"

