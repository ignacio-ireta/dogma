"""Public tests: the examples from problem.md, duplicates, the empty input, and a NaN.

The learner's code is imported from submission.py. Tests never import solution.py.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from submission import running_median


def _solve(values: Any) -> list[float]:
    try:
        got = running_median(values)
    except Exception as exc:
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check the heap sizes and the even-count mean"
        )
    assert isinstance(got, list) and len(got) == len(values), (
        "return one median per prefix, as a list of the same length as the input"
    )
    return got


def test_prefix_medians_for_mixed_insertions() -> None:
    got = _solve([2, 1, 3, 4])
    assert all(
        math.isclose(a, b, rel_tol=1e-12) for a, b in zip(got, [2.0, 1.5, 2.0, 2.5], strict=True)
    ), "after 1 value the median is 2; after 2 values it is the mean 1.5; after 4 it is 2.5"


def test_duplicates_keep_the_median_at_that_value() -> None:
    assert _solve([7, 7, 7]) == [7.0, 7.0, 7.0], (
        "repeated values give that value as the median at every step"
    )


def test_empty_input_gives_an_empty_list() -> None:
    assert _solve([]) == [], "no values give no medians"


def test_nan_raises_value_error() -> None:
    try:
        running_median([1.0, float("nan")])
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"NaN must raise ValueError (raised {type(exc).__name__})")
    pytest.fail("NaN was accepted: raise ValueError, since NaN has no place in an order")
