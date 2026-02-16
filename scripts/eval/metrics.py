from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List, Sequence, TypeVar

__all__ = [
    "normalize_text",
    "edit_distance",
    "cer",
    "wer",
    "ngram_repeat_ratio",
]

_T = TypeVar("_T")


_WHITESPACE_RE = re.compile(r"\s+")
_PUNC_TRANSLATION = str.maketrans({c: " " for c in string.punctuation})


def normalize_text(
    text: str,
    *,
    lowercase: bool = False,
    remove_punc: bool = False,
    remove_whitespace: bool = False,
    collapse_whitespace: bool = True,
) -> str:
    s = str(text or "")
    if lowercase:
        s = s.lower()
    if remove_punc:
        # Replace ASCII punctuation with spaces first so words don't get glued.
        s = s.translate(_PUNC_TRANSLATION)
        # Best-effort: also cover common CJK punctuation.
        for c in "，。！？；：、“”‘’（）《》【】…—·":
            s = s.replace(c, " ")
    if collapse_whitespace:
        s = _WHITESPACE_RE.sub(" ", s).strip()
    if remove_whitespace:
        s = s.replace(" ", "")
    return s


def edit_distance(ref: Sequence[_T], hyp: Sequence[_T]) -> int:
    """Levenshtein edit distance (insert/delete/substitute)."""
    if ref == hyp:
        return 0

    n = len(ref)
    m = len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n

    # O(min(n, m)) memory DP.
    if m < n:
        ref, hyp = hyp, ref
        n, m = m, n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i]
        ref_item = ref[i - 1]
        for j in range(1, m + 1):
            hyp_item = hyp[j - 1]
            sub_cost = 0 if ref_item == hyp_item else 1
            cur.append(
                min(
                    cur[j - 1] + 1,  # insertion
                    prev[j] + 1,  # deletion
                    prev[j - 1] + sub_cost,  # substitution
                )
            )
        prev = cur
    return prev[-1]


def cer(
    ref: str,
    hyp: str,
    *,
    lowercase: bool = False,
    remove_punc: bool = False,
    remove_whitespace: bool = False,
) -> float:
    """Character Error Rate."""
    ref_n = normalize_text(
        ref,
        lowercase=lowercase,
        remove_punc=remove_punc,
        remove_whitespace=remove_whitespace,
    )
    hyp_n = normalize_text(
        hyp,
        lowercase=lowercase,
        remove_punc=remove_punc,
        remove_whitespace=remove_whitespace,
    )

    ref_chars = list(ref_n)
    hyp_chars = list(hyp_n)
    denom = len(ref_chars)
    if denom == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return float(edit_distance(ref_chars, hyp_chars)) / float(denom)


def wer(
    ref: str,
    hyp: str,
    *,
    lowercase: bool = False,
    remove_punc: bool = False,
) -> float:
    """Word Error Rate (naive whitespace tokenization)."""
    ref_n = normalize_text(ref, lowercase=lowercase, remove_punc=remove_punc, collapse_whitespace=True)
    hyp_n = normalize_text(hyp, lowercase=lowercase, remove_punc=remove_punc, collapse_whitespace=True)

    ref_words = ref_n.split() if ref_n else []
    hyp_words = hyp_n.split() if hyp_n else []
    denom = len(ref_words)
    if denom == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return float(edit_distance(ref_words, hyp_words)) / float(denom)


def ngram_repeat_ratio(text: str, *, n: int = 4) -> float:
    """Heuristic duplication ratio: repeated n-grams / total n-grams (char-level).

    - 0.0 means no obvious repeats (for this n).
    - Higher values indicate more overlap-style repetition (useful for chunk merge debugging).
    """
    if n <= 0:
        return 0.0
    s = normalize_text(text, remove_whitespace=True, collapse_whitespace=True)
    if len(s) < n:
        return 0.0

    grams: List[str] = [s[i : i + n] for i in range(0, len(s) - n + 1)]
    if not grams:
        return 0.0

    counts = Counter(grams)
    repeats = sum((c - 1) for c in counts.values() if c > 1)
    return float(repeats) / float(len(grams))

