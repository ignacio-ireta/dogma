from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def ndcg_at_k(
    true_relevance: Sequence[float] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    k: int | None = None,
) -> float:
    """NDCG at k for one query, with tied scores averaged over their possible orders.

    true_relevance: 1-D, non-negative, finite gains. scores: 1-D, finite; higher ranks earlier.
    k: None for the whole list, or an integer >= 1 (k > n means n). Gain = relevance, discount
    1 / log2(rank + 1). A tie group occupying positions a..b contributes its mean relevance times
    the sum of the discounts of those positions within k. IDCG is the same sum on the relevance
    sorted descending. Return 0.0 when IDCG is 0.

    Returns
    -------
    NDCG as a Python float in [0, 1].

    Raises
    ------
    ValueError
        For inputs that are not 1-D, unequal lengths, empty input, negative or non-finite
        relevance, NaN or infinite scores, or a k that is neither None nor an integer >= 1.
    """
    raise NotImplementedError
