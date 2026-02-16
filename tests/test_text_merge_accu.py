from src.core.text_processor.text_merge_accu import (
    chars_to_text,
    linear_chars_with_timestamps,
    merge_chars_by_sequence_matcher,
)


def test_text_merge_accu_sequence_matcher_dedupes_overlap():
    # Simulate two overlapped chunks:
    # - chunk1: 0s..10s
    # - chunk2: 8s..18s (2s overlap)
    c1, t1 = linear_chars_with_timestamps("今天天气真好啊", start_s=0.0, end_s=10.0)
    c2, t2 = linear_chars_with_timestamps("天气真好啊我们出去玩", start_s=8.0, end_s=18.0)

    merged_c, merged_t = merge_chars_by_sequence_matcher(
        c1,
        t1,
        c2,
        t2,
        offset_s=8.0,
        overlap_s=2.0,
        is_first_segment=False,
    )

    assert len(merged_c) == len(merged_t)
    assert chars_to_text(merged_c) == "今天天气真好啊我们出去玩"


def test_text_merge_accu_fallback_time_dedupe_keeps_nonoverlap_tail():
    c1, t1 = linear_chars_with_timestamps("AAA", start_s=0.0, end_s=3.0)
    c2, t2 = linear_chars_with_timestamps("BBB", start_s=2.0, end_s=5.0)

    merged_c, merged_t = merge_chars_by_sequence_matcher(
        c1,
        t1,
        c2,
        t2,
        offset_s=2.0,
        overlap_s=1.0,
        is_first_segment=False,
        min_match_chars=10,  # force fallback
    )

    # The fallback should at least keep the tail of the second chunk.
    assert chars_to_text(merged_c).endswith("BB")


def test_text_merge_accu_cleanup_removes_double_spaces_and_punc():
    c1, t1 = linear_chars_with_timestamps("Hi  ", start_s=0.0, end_s=1.0)
    c2, t2 = linear_chars_with_timestamps("  !!", start_s=0.8, end_s=2.0)

    merged_c, merged_t = merge_chars_by_sequence_matcher(
        c1,
        t1,
        c2,
        t2,
        offset_s=0.8,
        overlap_s=0.2,
        is_first_segment=False,
        min_match_chars=10,  # force fallback
    )

    text = chars_to_text(merged_c)
    assert "  " not in text
    assert "!!" not in text

