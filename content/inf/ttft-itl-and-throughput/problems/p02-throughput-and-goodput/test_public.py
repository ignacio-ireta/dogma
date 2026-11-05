"""Public tests: the worked examples from problem.md, and the zero-window rule."""

from __future__ import annotations

from typing import Any

import pytest
from submission import serving_rates

WORKED = [
    {"arrival": 0.0, "token_times": [0.5, 0.6, 0.8]},
    {"arrival": 0.2, "token_times": [1.2]},
    {"arrival": 1.0, "token_times": [1.1, 1.3, 1.6, 2.0]},
]


def _pair(got: Any, want: tuple[float, float], hint: str = "") -> None:
    if type(got) is not tuple or len(got) != 2:
        pytest.fail("return a tuple of (throughput, goodput)")
    if type(got[0]) is not float or type(got[1]) is not float:
        pytest.fail("throughput and goodput must be Python floats")
    if abs(got[0] - want[0]) > 1e-9:
        pytest.fail(
            "throughput is wrong: count every token, including first tokens, over a window "
            "that starts at the earliest arrival"
        )
    if abs(got[1] - want[1]) > 1e-9:
        pytest.fail(hint or "goodput is wrong: count only requests meeting both inclusive bounds")


def test_worked_example_counts_tokens_and_only_the_passing_request() -> None:
    _pair(serving_rates(WORKED, 0.6, 0.2), (4.0, 0.5))


def test_bounds_are_inclusive_on_both_slos() -> None:
    requests = [
        {"arrival": 0.0, "token_times": [0.5, 1.0]},
        {"arrival": 0.0, "token_times": [1.0]},
    ]
    _pair(serving_rates(requests, 0.5, 0.5), (3.0, 1.0))


def test_zero_length_window_is_rejected() -> None:
    try:
        serving_rates([{"arrival": 1.0, "token_times": [1.0]}], 0.5, 0.5)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"a zero-length window must raise ValueError, instead it raised {type(exc).__name__}"
        )
    pytest.fail("a zero-length window must raise ValueError: there is no duration to divide by")
