from __future__ import annotations


def top_k_frequent(tokens: list[str], k: int) -> list[tuple[str, int]]:
    """Return the ``k`` most frequent tokens, ties broken alphabetically."""
    # Step 1, the counting pattern: one pass, one hash map. A dict lookup or insert costs O(1) on
    # average, so this pass is O(n) for n tokens. dict.get supplies 0 on a first occurrence, which
    # avoids a KeyError.
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    # Step 2: guard k <= 0 before slicing. A negative slice end such as ranked[:-2] would silently
    # return the wrong rows instead of an empty list.
    if k <= 0:
        return []

    # Step 3: rank the d distinct tokens. Sorting by (-count, token) puts the highest count first
    # and breaks ties alphabetically, in a single sort. Sorting costs O(d log d). A heap of size k
    # would cost O(d log k), which is cheaper when k is much smaller than d (see the lesson).
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    # Step 4: slicing handles k larger than the vocabulary. The slice then returns every entry.
    return ranked[:k]
