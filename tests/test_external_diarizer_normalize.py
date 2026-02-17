from src.core.speaker.external_diarizer_normalize import normalize_segments


def test_normalize_segments_sorts_clamps_and_drops_invalid():
    raw = [
        {"spk": 1, "start": 2000, "end": 1000},  # invalid (end < start) -> drop
        {"spk": 0, "start": -5, "end": 10},      # clamp start to 0
        {"spk": 0, "start": 10, "end": 20},
    ]
    segs = normalize_segments(raw, duration_ms=15)
    assert segs == [
        {"spk": 0, "start": 0, "end": 10},
        {"spk": 0, "start": 10, "end": 15},
    ]
