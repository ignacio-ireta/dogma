from __future__ import annotations


def longest_window_within_budget(costs: list[int], budget: int) -> tuple[int, int]:
    """Wrong on purpose: check every start and extend each window to the right."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if any(cost < 0 for cost in costs):
        raise ValueError("costs must be non-negative")
    best_length, best_start = 0, 0
    n = len(costs)
    for start in range(n):
        total = 0
        for end in range(start, n):
            total += costs[end]
            if total > budget:
                break
            length = end - start + 1
            if length > best_length:
                best_length, best_start = length, start
    return best_length, best_start
