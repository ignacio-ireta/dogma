from __future__ import annotations


def longest_common_subsequence(a: str, b: str) -> tuple[int, str]:
    """Wrong on purpose: the witness is returned in reverse order. Return (length, witness): the length of a longest common subsequence of a and b, and one
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
    """  # noqa: E501
    if type(a) is not str or type(b) is not str:
        raise TypeError(
            f"a and b must be Python str, got {type(a).__name__} and {type(b).__name__}"
        )

    n, m = len(a), len(b)
    # State: dp[i][j] is the LCS length of the prefixes a[:i] and b[:j].
    # Base case: dp[0][j] = dp[i][0] = 0, because an empty prefix has no common characters.
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # Order of evaluation: dp[i][j] reads only dp[i-1][j-1], dp[i-1][j] and dp[i][j-1], all
    # computed earlier when i and j increase.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                # Transition 1: the last characters match, so the LCS can use them both.
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # Transition 2: they differ, so drop one of them and keep the better result.
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Answer location: dp[n][m] covers the whole of both strings.
    length = dp[n][m]

    # Reconstruction: walk back from (n, m) and follow the transition that produced each value.
    # A match is always safe to take, so it is taken. A mismatch moves toward the larger value.
    i, j = n, m
    pieces: list[str] = []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            pieces.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # The walk collects characters from the end, so reverse them to restore the order.
    witness = "".join(pieces)
    return length, witness
