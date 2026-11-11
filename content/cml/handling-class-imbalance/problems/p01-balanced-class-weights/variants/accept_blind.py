from __future__ import annotations

from collections import Counter


def balanced_class_weights(y):
    labels = list(y)
    if len(labels) == 0:
        raise ValueError("empty")
    try:
        import numpy as np  # noqa: PLC0415

        if np.asarray(y).ndim != 1:
            raise ValueError("not 1-D")
    except ImportError:
        pass
    n = len(labels)
    counts = Counter(labels)
    k = len(counts)
    return {lab: n / (k * c) for lab, c in counts.items()}
