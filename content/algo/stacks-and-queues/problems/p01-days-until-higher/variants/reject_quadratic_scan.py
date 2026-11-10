from __future__ import annotations


def days_until_higher(values: list[float]) -> list[int]:
    """Wrong on purpose: for each index, scan every later position until a larger reading is found."""  # noqa: E501
    result = [0] * len(values)
    for index, value in enumerate(values):
        for later in range(index + 1, len(values)):
            if values[later] > value:
                result[index] = later - index
                break
    return result
