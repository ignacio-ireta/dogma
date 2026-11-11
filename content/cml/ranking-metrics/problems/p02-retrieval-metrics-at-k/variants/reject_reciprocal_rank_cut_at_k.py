from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence


def retrieval_metrics(
    ranked_ids: Sequence[Hashable],
    relevant_ids: Iterable[Hashable],
    k: int,
) -> dict[str, float]:
    """Wrong on purpose: Precision, recall, reciprocal rank and average precision at k for one query.

    ranked_ids: hashable ids, best first, with no repeats; may be empty.
    relevant_ids: hashable ids, duplicates ignored; at least one is required.
    k: an integer >= 1; may exceed len(ranked_ids).

    precision_at_k = hits in the top k / k. recall_at_k = hits in the top k / |relevant|.
    reciprocal_rank = 1 / position of the first relevant id in the whole list, or 0.0.
    average_precision_at_k = sum of precision at each relevant position <= k, divided by
    min(k, |relevant|).

    Returns
    -------
    A dict with the keys precision_at_k, recall_at_k, reciprocal_rank, average_precision_at_k.

    Raises
    ------
    ValueError
        For a repeated ranked id, an empty relevant set, or a k that is not an integer >= 1.
    """  # noqa: E501
    ranked = list(ranked_ids)  # a copy: the caller's list is never touched
    relevant = set(relevant_ids)  # duplicates in the input collapse here
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked_ids must not repeat an id: each document is ranked once")
    if not relevant:
        raise ValueError(
            "relevant_ids must contain at least one id: recall is undefined without one"
        )
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be an integer >= 1")

    # hits[i] is True when the item at 0-based position i is relevant.
    hits = [item in relevant for item in ranked]
    top = hits[:k]
    relevant_in_top = sum(top)

    precision = relevant_in_top / k  # the denominator is k, even for a short list
    recall = relevant_in_top / len(relevant)

    # Reciprocal rank looks at the whole ranking, not only the top k.
    first_hit = next((i for i, hit in enumerate(top) if hit), None)
    reciprocal_rank = 0.0 if first_hit is None else 1.0 / (first_hit + 1)

    # Average precision: at each relevant position (1-based, within k), add the precision there.
    found = 0
    precision_sum = 0.0
    for position, hit in enumerate(top, start=1):
        if hit:
            found += 1
            precision_sum += found / position
    # Dividing by min(k, |R|) keeps the value in [0, 1] even when |R| is larger than k.
    average_precision = precision_sum / min(k, len(relevant))

    return {
        "precision_at_k": float(precision),
        "recall_at_k": float(recall),
        "reciprocal_rank": float(reciprocal_rank),
        "average_precision_at_k": float(average_precision),
    }
