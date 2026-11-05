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
    raise NotImplementedError
