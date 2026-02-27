"""快速 RAG 检索模块

This module uses Numba JIT to accelerate fuzzy substring distance over
phoneme-id sequences. In some environments (notably certain ONNX runtime
images), the Numba stack can be present but broken due to binary or version
incompatibilities (e.g. NumPy/Numba mismatch). That must not crash the API.

We therefore treat the JIT path as best-effort and fall back to a pure-Python
implementation when the JIT path errors.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    # Keep runtime resilient even if numba/llvmlite cannot be imported.
    NUMBA_AVAILABLE = False

    def njit(*_args, **_kwargs):
        def _decorator(fn):
            return fn

        return _decorator

from src.core.hotword.phoneme import Phoneme

@njit(cache=True)
def _fuzzy_substring_numba(main, sub):
    """Numba 加速的模糊子串距离计算"""
    n, m = len(sub), len(main)
    if n == 0 or m == 0:
        return float(n)

    dp = np.zeros((n+1, m+1), dtype=np.float32)
    for i in range(1, n+1):
        dp[i, 0] = float(i)

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0.0 if sub[i-1] == main[j-1] else 1.0
            dp[i, j] = min(dp[i-1, j] + 1.0,
                          dp[i, j-1] + 1.0,
                          dp[i-1, j-1] + cost)

    return np.min(dp[n, 1:])


class FastRAG:
    """快速 RAG 热词检索"""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.ph_to_id: Dict[str, int] = {}
        self.index: Dict[int, List[Tuple[str, any]]] = defaultdict(list)
        self.hotword_count = 0
        self._numba_enabled = bool(NUMBA_AVAILABLE)
        self._numba_error_logged = False

    def _encode(self, phs: List[Phoneme]):
        """将音素序列编码为整数序列"""
        ids = []
        for p in phs:
            if p.value not in self.ph_to_id:
                self.ph_to_id[p.value] = len(self.ph_to_id) + 1
            ids.append(self.ph_to_id[p.value])
        return np.array(ids, dtype=np.int32)


    def add_hotwords(self, hotwords: Dict[str, List[Phoneme]]):
        """添加热词到索引"""
        for hw, phs in hotwords.items():
            if not phs:
                continue
            codes = self._encode(phs)
            # 使用前两个音素建立倒排索引
            for i in range(min(len(codes), 2)):
                self.index[codes[i]].append((hw, codes))
            self.hotword_count += 1

    def search(self, input_phs: List[Phoneme], top_k: int = 10) -> List[Tuple[str, float]]:
        """搜索匹配的热词"""
        if not input_phs:
            return []

        input_codes = self._encode(input_phs)
        unique = set(input_codes.tolist())
        candidates = []
        for c in unique:
            candidates.extend(self.index.get(c, []))

        seen = set()
        results = []

        for hw, cands in candidates:
            if hw in seen or len(cands) > len(input_codes) + 3:
                continue
            seen.add(hw)
            dist = None
            if self._numba_enabled:
                try:
                    dist = _fuzzy_substring_numba(input_codes, cands)
                except Exception as e:
                    # If JIT fails at runtime, disable it for the rest of this
                    # process to avoid crashing transcription requests.
                    self._numba_enabled = False
                    if not self._numba_error_logged:
                        self._numba_error_logged = True
                        msg = (
                            f"FastRAG numba path failed and is now disabled: {e}. "
                            "Falling back to pure-Python distance (slower)."
                        )
                        # Import logger lazily to avoid circular imports.
                        import logging

                        logging.getLogger(__name__).warning(msg)

            if dist is None:
                dist = self._python_dist(input_codes, cands)
            score = 1.0 - (dist / len(cands))
            if score >= self.threshold:
                results.append((hw, round(float(score), 3)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _python_dist(self, main, sub) -> float:
        """纯 Python 实现的模糊子串距离"""
        n, m = len(sub), len(main)
        if m == 0:
            return float(n)
        dp = [[0.0] * (m+1) for _ in range(n+1)]

        for i in range(1, n+1):
            dp[i][0] = float(i)

        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0.0 if sub[i-1] == main[j-1] else 1.0
                dp[i][j] = min(dp[i-1][j] + 1.0,
                              dp[i][j-1] + 1.0,
                              dp[i-1][j-1] + cost)

        return min(dp[n][1:]) if m > 0 else float(n)
