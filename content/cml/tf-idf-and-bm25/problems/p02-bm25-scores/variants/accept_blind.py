from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def bm25_scores(
    query_tokens: Sequence[str],
    docs_tokens: Sequence[Sequence[str]],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[float]:
    if len(docs_tokens) == 0:
        raise ValueError("docs_tokens must not be empty")
    if not (k1 >= 0) or not math.isfinite(k1):
        raise ValueError("k1 must be non-negative")
    if not (0.0 <= b <= 1.0):
        raise ValueError("b must be in [0, 1]")
    N = len(docs_tokens)
    counters = [Counter(d) for d in docs_tokens]
    lengths = [len(d) for d in docs_tokens]
    avgdl = sum(lengths) / N
    if avgdl == 0 or len(query_tokens) == 0:
        return [0.0] * N
    qcount = Counter(query_tokens)
    idf = {}
    for q in qcount:
        n = sum(1 for c in counters if q in c)
        idf[q] = math.log(1.0 + (N - n + 0.5) / (n + 0.5))
    scores = []
    for c, dl in zip(counters, lengths):  # noqa: B905
        s = 0.0
        for q, m in qcount.items():
            f = c.get(q, 0)
            if f:
                s += m * idf[q] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(float(s))
    return scores
