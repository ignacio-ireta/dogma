from __future__ import annotations


def top_k_frequent(tokens: list[str], k: int) -> list[tuple[str, int]]:
    """Wrong on purpose: folds case, so "The" and "the" count as one token.

    Return the ``k`` most frequent tokens, ties broken alphabetically.
    """
    counts: dict[str, int] = {}
    for token in tokens:
        key = token.lower()
        counts[key] = counts.get(key, 0) + 1
    if k <= 0:
        return []
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:k]
