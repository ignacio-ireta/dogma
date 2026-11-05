"""Public tests: the two worked examples from problem.md, and one validation rule."""

from __future__ import annotations

from typing import Any

import pytest
from submission import latency_summary

WORKED = [
    {"arrival": 0.0, "token_times": [0.5, 0.6, 0.8]},
    {"arrival": 0.2, "token_times": [1.2]},
    {"arrival": 1.0, "token_times": [1.1, 1.3, 1.6, 2.0]},
]


def _close(got: Any, expected: float, mistake: str) -> None:
    if type(got) is not float:
        pytest.fail(f"{mistake}: return a Python float, got {type(got).__name__}")
    if abs(got - expected) > 1e-9:
        pytest.fail(mistake)


def test_worked_example_percentiles_use_linear_interpolation() -> None:
    summary = latency_summary(WORKED)
    _close(summary["ttft_p50"], 0.5, "the TTFT median is wrong: TTFT is first token minus arrival")
    _close(
        summary["ttft_p95"], 0.95, "the TTFT p95 is wrong: interpolate between the two top values"
    )
    _close(summary["itl_p50"], 0.225, "the ITL median is wrong: average each request's n - 1 gaps")
    _close(
        summary["itl_p95"],
        0.2925,
        "the ITL p95 is wrong: a p95 interpolates the sorted values, not the mean",
    )
    _close(
        summary["e2e_p50"],
        1.0,
        "the end-to-end median is wrong: count from arrival to the last token",
    )
    _close(
        summary["e2e_p95"], 1.0, "the end-to-end p95 is wrong: count from arrival to the last token"
    )


def test_one_token_request_has_no_itl() -> None:
    summary = latency_summary([{"arrival": 0.0, "token_times": [0.25]}])
    assert summary["itl_p50"] is None and summary["itl_p95"] is None, (
        "a one-token request has no gaps: the ITL values must be None, not 0"
    )
    _close(summary["ttft_p95"], 0.25, "a single request's p95 must be its only value")


def test_empty_request_list_is_rejected() -> None:
    try:
        latency_summary([])
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"an empty list must raise ValueError, instead it raised {type(exc).__name__}")
    pytest.fail("an empty list must raise ValueError: there is nothing to summarise")
