from __future__ import annotations


def longest_window_within_budget(costs: list[int], budget: int) -> tuple[int, int]:
    """Wrong on purpose: return ``(length, start)`` of the longest contiguous window with sum <= budget."""  # noqa: E501
    # Validate first. The two-pointer invariant below relies on costs being non-negative, so a
    # negative entry must fail loudly rather than produce a plausible wrong answer.
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if any(cost < 0 for cost in costs):
        raise ValueError("costs must be non-negative")

    # Invariant: costs[left:right + 1] is the window under consideration, and window_sum is its sum.
    # Because every cost is >= 0, extending the window to the right never lowers the sum, and
    # shrinking it from the left never raises the sum. So for each right end, the valid windows
    # form one block, and "shrink from the left until valid" keeps the longest valid window that
    # ends at right.
    best_length, best_start = 0, 0
    left = 0
    window_sum = 0
    for right, cost in enumerate(costs[:-1]):
        # Grow: add the new entry on the right.
        window_sum += cost
        # Shrink: remove entries from the left until the window is valid again. This is a while
        # loop, not an if, because one large entry can force several removals. If the entry alone
        # is above the budget, left moves past it and the window becomes empty (length 0).
        while window_sum > budget:
            window_sum -= costs[left]
            left += 1
        # Each index enters the window once and leaves it at most once, so the total work is O(n).
        length = right - left + 1
        # Strictly greater keeps the first window of each maximum length. Right ends are visited in
        # increasing order, so the first one found has the smallest start.
        if length > best_length:
            best_length, best_start = length, left
    return best_length, best_start
