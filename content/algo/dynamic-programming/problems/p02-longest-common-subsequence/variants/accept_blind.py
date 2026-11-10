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
    if not isinstance(a, str) or not isinstance(b, str):
        raise TypeError("a and b must be Python str")

    rows, cols = len(a), len(b)
    # table[i][j] = LCS length of a[:i] and b[:j]
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    # Walk back from the end to recover one longest subsequence.
    pieces: list[str] = []
    i, j = rows, cols
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            pieces.append(a[i - 1])
            i -= 1
            j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    witness = "".join(reversed(pieces))
    return len(witness), witness
