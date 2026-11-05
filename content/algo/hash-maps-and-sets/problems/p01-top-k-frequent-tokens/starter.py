from __future__ import annotations


def top_k_frequent(tokens: list[str], k: int) -> list[tuple[str, int]]:
    """Return the ``k`` most frequent tokens as ``(token, count)`` pairs.

    Parameters
    ----------
    tokens : list of str, possibly empty. Compared exactly (case-sensitive).
    k : int. ``k <= 0`` returns ``[]``. A ``k`` larger than the number of distinct tokens
        returns all of them.

    Returns
    -------
    list of ``(token, count)`` tuples, sorted by count descending, then token ascending
    (Python string order). Length ``min(max(k, 0), number of distinct tokens)``.
    """
    raise NotImplementedError
