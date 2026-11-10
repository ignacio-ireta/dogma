from __future__ import annotations


def top_k_frequent(tokens: list[str], k: int) -> list[tuple[str, int]]:
    """Wrong on purpose: skips the empty string, which is an ordinary token.

    Return the ``k`` most frequent tokens, ties broken alphabetically.
    """
    counts: dict[str, int] = {}
    for token in tokens:
        if not token:
            continue
        counts[token] = counts.get(token, 0) + 1
    if k <= 0:
        return []
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:k]
