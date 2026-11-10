from __future__ import annotations


def days_until_higher(values: list[float]) -> list[int]:
    """Wrong on purpose: return, for each index, the steps to the next strictly larger reading (0 if none)."""  # noqa: E501
    # A monotonic stack of indices. Each index waits on the stack until a strictly larger reading
    # arrives. The values of the waiting indices never increase from bottom to top, because a new
    # reading pops every smaller waiting reading before it is pushed.
    if not values:
        return [0]
    result = [0] * len(values)
    waiting: list[int] = []
    for index, value in enumerate(values):
        # The current reading is the first later reading that is strictly larger than each popped
        # index's value, so its wait ends here, and the wait is the distance between the two
        # indices.
        # A while loop, not an if: one reading can end the waits of several earlier readings.
        # Equal readings are not popped (the test is strict), so they keep waiting.
        while waiting and values[waiting[-1]] < value:
            earlier = waiting.pop()
            result[earlier] = index - earlier
        waiting.append(index)
    # Indices still on the stack never saw a strictly larger reading. Their entries stay 0.
    # Each index is pushed once and popped at most once, so the loop does O(n) work in total.
    return result
