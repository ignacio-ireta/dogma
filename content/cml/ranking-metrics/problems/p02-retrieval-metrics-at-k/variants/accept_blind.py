from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence


def retrieval_metrics(
    ranked_ids: Sequence[Hashable],
    relevant_ids: Iterable[Hashable],
    k: int,
) -> dict[str, float]:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be an integer >= 1")
    ranked = list(ranked_ids)
    if len(set(ranked)) != len(ranked):
        raise ValueError("repeated id")
    R = set(relevant_ids)
    if not R:
        raise ValueError("no relevant ids")
    n = len(ranked)  # noqa: F841
    hits = 0
    ap_sum = 0.0
    for i, item in enumerate(ranked[:k], start=1):
        if item in R:
            hits += 1
            ap_sum += hits / i
    rr = 0.0
    for i, item in enumerate(ranked, start=1):
        if item in R:
            rr = 1.0 / i
            break
    return {
        "precision_at_k": float(hits / k),
        "recall_at_k": float(hits / len(R)),
        "reciprocal_rank": float(rr),
        "average_precision_at_k": float(ap_sum / min(k, len(R))),
    }
