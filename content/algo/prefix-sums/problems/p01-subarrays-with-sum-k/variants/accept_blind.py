from __future__ import annotations


def count_subarrays_with_sum(nums: list[int], k: int) -> int:
    """Return how many contiguous, non-empty subarrays of nums sum to exactly k."""
    if type(k) is not int:
        raise TypeError("k must be a Python int")
    for value in nums:
        if type(value) is not int:
            raise TypeError("every entry of nums must be a Python int")

    seen: dict[int, int] = {0: 1}
    prefix = 0
    total = 0
    for value in nums:
        prefix += value
        total += seen.get(prefix - k, 0)
        seen[prefix] = seen.get(prefix, 0) + 1
    return total
