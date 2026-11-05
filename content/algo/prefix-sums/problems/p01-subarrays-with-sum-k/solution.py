from __future__ import annotations


def count_subarrays_with_sum(nums: list[int], k: int) -> int:
    """Return how many contiguous, non-empty subarrays of nums sum to exactly k.

    Parameters
    ----------
    nums : list of Python int, possibly empty. Negative, zero and positive values are allowed.
        Must not be mutated.
    k : Python int, possibly negative or zero.

    Returns
    -------
    int : the number of index pairs (i, j) with 0 <= i < j <= len(nums) and sum(nums[i:j]) == k.
        Runs in O(n) time for n = len(nums). Subarrays are counted by position, not by value.

    Raises
    ------
    TypeError
        When k or any entry of nums is not a Python int (bool, float, str, None and NumPy
        integers are all rejected).
    """
    # Validate first, so a bad entry is named before any arithmetic happens. type(...) is int
    # rejects bool, which is a subclass of int, and NumPy integers, which are not Python ints.
    if type(k) is not int:
        raise TypeError(f"k must be a Python int, got {type(k).__name__}")
    for value in nums:
        if type(value) is not int:
            raise TypeError(f"every entry must be a Python int, got {type(value).__name__}")

    # Step 1: prefix sums. P_j is the sum of the first j entries, with P_0 = 0. The sum of
    # nums[i:j] is P_j - P_i, so nums[i:j] sums to k exactly when P_i = P_j - k.
    # Python ints are exact, so no rounding can hide a subarray here.
    #
    # Step 2: count, for each j, the earlier prefixes equal to P_j - k. counts holds the
    # prefixes seen so far, and it starts with P_0 = 0 (the empty prefix). Without that seed,
    # every subarray that starts at position 0 is missed.
    counts: dict[int, int] = {0: 1}
    prefix = 0
    total = 0
    for value in nums:
        prefix += value
        # Look up before inserting P_j. With k = 0 this matters: if P_j were inserted first,
        # the lookup would find P_j itself, which counts the empty subarray nums[j:j].
        total += counts.get(prefix - k, 0)
        counts[prefix] = counts.get(prefix, 0) + 1
    return total
