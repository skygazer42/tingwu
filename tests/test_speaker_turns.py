from src.core.speaker.turns import build_speaker_turns


def test_build_speaker_turns_merges_consecutive_same_speaker_with_small_gap():
    sentences = [
        {"speaker": "说话人1", "speaker_id": 0, "start": 0, "end": 1000, "text": "你好"},
        {"speaker": "说话人1", "speaker_id": 0, "start": 1100, "end": 2000, "text": "今天"},
    ]

    turns = build_speaker_turns(sentences, gap_ms=800)

    assert len(turns) == 1
    assert turns[0]["speaker"] == "说话人1"
    assert turns[0]["speaker_id"] == 0
    assert turns[0]["start"] == 0
    assert turns[0]["end"] == 2000
    assert turns[0]["text"] == "你好今天"
    assert turns[0]["sentence_count"] == 2


def test_build_speaker_turns_does_not_merge_across_speaker_change():
    sentences = [
        {"speaker": "说话人1", "speaker_id": 0, "start": 0, "end": 1000, "text": "A"},
        {"speaker": "说话人2", "speaker_id": 1, "start": 1050, "end": 2000, "text": "B"},
    ]

    turns = build_speaker_turns(sentences, gap_ms=800)

    assert len(turns) == 2
    assert turns[0]["speaker"] == "说话人1"
    assert turns[1]["speaker"] == "说话人2"


def test_build_speaker_turns_does_not_merge_when_gap_exceeds_threshold():
    sentences = [
        {"speaker": "说话人1", "speaker_id": 0, "start": 0, "end": 1000, "text": "A"},
        {"speaker": "说话人1", "speaker_id": 0, "start": 2000, "end": 3000, "text": "B"},
    ]

    turns = build_speaker_turns(sentences, gap_ms=800)

    assert len(turns) == 2
    assert turns[0]["text"] == "A"
    assert turns[1]["text"] == "B"


def test_build_speaker_turns_missing_timestamps_default_to_zero():
    turns = build_speaker_turns([{"speaker": "说话人1", "speaker_id": 0, "text": "A"}])
    assert len(turns) == 1
    assert turns[0]["start"] == 0
    assert turns[0]["end"] == 0


def test_build_speaker_turns_drops_empty_text_by_default_min_chars():
    turns = build_speaker_turns([{"speaker": "说话人1", "speaker_id": 0, "start": 0, "end": 1000, "text": ""}])
    assert turns == []


def test_build_speaker_turns_unknown_speaker_id_defaults_to_minus_one():
    turns = build_speaker_turns([{"speaker": "未知", "speaker_id": "wat", "start": 0, "end": 1000, "text": "A"}])
    assert len(turns) == 1
    assert turns[0]["speaker_id"] == -1
