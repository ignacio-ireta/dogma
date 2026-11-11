from __future__ import annotations

import math

import numpy as np


def ndcg_at_k(true_relevance, scores, k=None) -> float:
    rel = np.asarray(true_relevance, dtype=float)
    s = np.asarray(scores, dtype=float)
    if rel.ndim != 1 or s.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    if len(rel) != len(s) or len(rel) == 0:
        raise ValueError("lengths differ or input is empty")
    if not np.isfinite(rel).all() or (rel < 0).any():
        raise ValueError("relevance must be finite and non-negative")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if k is not None and (not isinstance(k, int) or k < 1):
        raise ValueError("k must be None or an integer >= 1")
    n = len(rel)
    cutoff = n if k is None else min(k, n)
    groups: dict[float, list[int]] = {}
    for i, v in enumerate(s.tolist()):
        groups.setdefault(v, []).append(i)
    dcg = 0.0
    position = 0
    for value in sorted(groups, reverse=True):
        members = groups[value]
        size = len(members)
        mean_gain = float(np.mean(rel[members]))
        mass = sum(1.0 / math.log2(r + 2) for r in range(position, position + size) if r < cutoff)
        dcg += mean_gain * mass
        position += size
    ideal = sorted(rel.tolist(), reverse=True)
    idcg = sum(ideal[r] / math.log2(r + 2) for r in range(cutoff))
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)
