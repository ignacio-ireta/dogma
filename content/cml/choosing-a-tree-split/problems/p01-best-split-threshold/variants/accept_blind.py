import math
from collections import Counter

import numpy as np


def _impurity(counts, n, criterion):
    if n == 0:
        return 0.0
    if criterion == "gini":
        return 1.0 - sum((c / n) ** 2 for c in counts.values())
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def best_split(x, y, *, criterion="gini"):
    if criterion not in ("gini", "entropy"):
        raise ValueError("bad criterion")
    xs = np.asarray(x, dtype=float)
    ys = list(y)
    if xs.ndim != 1 or xs.size == 0 or not np.isfinite(xs).all():
        raise ValueError("bad x")
    if len(ys) != xs.size:
        raise ValueError("bad y")
    n = xs.size
    order = sorted(range(n), key=lambda i: xs[i])
    total = Counter(ys)
    parent = _impurity(total, n, criterion)
    left = Counter()
    best_t, best_d = None, None
    for pos in range(n - 1):
        left[ys[order[pos]]] += 1
        a, b = xs[order[pos]], xs[order[pos + 1]]
        if not a < b:
            continue  # no candidate between equal values
        right = total - left
        nl, nr = pos + 1, n - pos - 1
        d = (
            parent
            - nl / n * _impurity(left, nl, criterion)
            - nr / n * _impurity(right, nr, criterion)
        )
        t = a / 2.0 + b / 2.0
        if best_t is None or d > best_d + 1e-12:
            best_t, best_d = float(t), float(d)
    if best_t is None:
        return None, 0.0
    return best_t, best_d
