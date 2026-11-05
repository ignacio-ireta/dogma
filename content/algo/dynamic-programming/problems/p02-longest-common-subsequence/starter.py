from __future__ import annotations


def longest_common_subsequence(a: str, b: str) -> tuple[int, str]:
    """Return (length, witness): the length of a longest common subsequence of a and b, and one
    such subsequence as a string of that length.

    Parameters
    ----------
    a, b : Python str, possibly empty. Neither is mutated.

    Returns
    -------
    tuple[int, str] : (length, witness). Empty input gives (0, "").
        Runs in O(len(a) * len(b)) time.

    Raises
    ------
    TypeError
        When a or b is not a Python str.
    """
    raise NotImplementedError
