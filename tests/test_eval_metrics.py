import math

from scripts.eval.metrics import cer, normalize_text, ngram_repeat_ratio, wer


def test_cer_zero_for_equal_text():
    assert cer("你好世界", "你好世界") == 0.0


def test_cer_one_insertion():
    # ref length=3, insert one char -> distance=1 => CER=1/3
    assert math.isclose(cer("abc", "abxc"), 1 / 3, rel_tol=0, abs_tol=1e-9)


def test_wer_simple_deletion():
    # ref words=2, delete one word -> distance=1 => WER=1/2
    assert math.isclose(wer("hello world", "hello"), 0.5, rel_tol=0, abs_tol=1e-9)


def test_normalize_text_remove_punc_and_collapse_whitespace():
    out = normalize_text(" Hello,   world! \n", lowercase=True, remove_punc=True)
    assert out == "hello world"


def test_ngram_repeat_ratio_detects_repeats():
    assert ngram_repeat_ratio("abcdefg", n=3) == 0.0
    assert ngram_repeat_ratio("abcabcabc", n=3) > 0.0

