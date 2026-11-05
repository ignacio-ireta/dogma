from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence


def retrieval_metrics(
    ranked_ids: Sequence[Hashable],
    relevant_ids: Iterable[Hashable],
    k: int,
) -> dict[str, float]:
    """Precision, recall, reciprocal rank and average precision at k for one query.

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
    """
    raise NotImplementedError
