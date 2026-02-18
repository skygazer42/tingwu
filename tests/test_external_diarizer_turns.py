from src.core.speaker.external_diarizer_turns import segments_to_turns


def test_segments_to_turns_merges_by_speaker_and_gap():
    segs = [
        {"spk": 0, "start": 0, "end": 1000},
        {"spk": 0, "start": 1100, "end": 2000},
        {"spk": 1, "start": 2100, "end": 3000},
    ]
    turns = segments_to_turns(segs, gap_ms=200, label_style="numeric")
    assert len(turns) == 2
    assert turns[0]["speaker_id"] == 0 and turns[0]["start"] == 0 and turns[0]["end"] == 2000
    assert turns[1]["speaker_id"] == 1

