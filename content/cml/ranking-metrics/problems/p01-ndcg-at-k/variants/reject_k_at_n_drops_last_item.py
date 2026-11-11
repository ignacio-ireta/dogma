from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def ndcg_at_k(
    true_relevance: Sequence[float] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    k: int | None = None,
) -> float:
    """Wrong on purpose: NDCG at k for one query, with tied scores averaged over their possible orders.

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
    """  # noqa: E501
    relevance = np.asarray(true_relevance, dtype=float)
    score = np.asarray(scores, dtype=float)
    if relevance.ndim != 1 or score.ndim != 1:
        raise ValueError("true_relevance and scores must be one-dimensional")
    if len(relevance) != len(score):
        raise ValueError("true_relevance and scores differ in length")
    if len(relevance) == 0:
        raise ValueError("at least one item is required")
    if not np.all(np.isfinite(relevance)) or np.any(relevance < 0):
        raise ValueError("relevance must be finite and non-negative")
    if not np.all(np.isfinite(score)):
        raise ValueError("scores must be finite: NaN and infinity have no place in the ranking")
    if k is not None and (not isinstance(k, int) or k < 1):
        raise ValueError("k must be None or an integer >= 1")

    n = len(relevance)
    cutoff = n if k is None else min(k, n - 1)

    # Discount of each 1-based position. Positions past the cutoff contribute nothing.
    positions = np.arange(1, n + 1)
    discount = np.where(positions <= cutoff, 1.0 / np.log2(positions + 1), 0.0)
    # prefix[r] is the total discount of positions 1..r, so a run's mass is one subtraction.
    prefix = np.concatenate([[0.0], np.cumsum(discount)])

    # Sort by score, descending. The stable sort only fixes the order of the gains inside a run;
    # the tie rule below does not depend on it.
    order = np.argsort(-score, kind="stable")
    sorted_scores = score[order]
    # A run is a maximal block of equal scores. It occupies 0-based positions first..last.
    run_start = np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
    run_id = np.cumsum(run_start) - 1
    first = np.flatnonzero(run_start)
    last = np.r_[first[1:], n] - 1
    counts = last - first + 1
    # Mean gain of each run: the expected gain at any position the run occupies.
    run_gain_sum = np.bincount(run_id, weights=relevance[order])
    mean_gain = run_gain_sum / counts
    # Each run contributes its mean gain times the discount mass of the positions it occupies.
    mass = prefix[last + 1] - prefix[first]
    dcg = float(np.sum(mean_gain * mass))

    # The ideal ordering sorts the relevance itself, so it has no ties to average.
    ideal = np.sort(relevance)[::-1]
    idcg = float(np.sum(ideal[:cutoff] * discount[:cutoff]))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
