from __future__ import annotations


def top_k_frequent(tokens: list[str], k: int) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    if k <= 0:
        return []
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [(token, count) for token, count in ranked[:k]]
