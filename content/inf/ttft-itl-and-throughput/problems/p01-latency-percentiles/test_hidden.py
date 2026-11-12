"""Hidden tests: an exact-fraction oracle on seeded requests, the definitions and validation rules.

Failure messages name the kind of mistake. They never print hidden timestamps or expected values.
A slow call is stopped by a signal after its time box and reported by name. The previous
signal handler is restored after each call.
"""

from __future__ import annotations

import copy
import functools
import math
import random
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from typing import Any

import pytest
from submission import latency_summary

from dogma_data import retail_invoices

KEYS = ("ttft_p50", "ttft_p95", "itl_p50", "itl_p95", "e2e_p50", "e2e_p95")
LARGE_N = 50_000
LARGE_CALL_S = 10.0
SMALL_CALL_S = 5.0


class _TooSlow(BaseException):
    """Raised inside the learner's function when a call runs past its time box."""


SLOW_MESSAGE = (
    "a call did not finish in time: rescanning the list for every rank is the usual cause; "
    "sort once, then read each percentile by position"
)
_HUNG: list[str] = []


@contextmanager
def _time_box(seconds: float) -> Iterator[None]:
    """Interrupt the learner's code after ``seconds``. Only the main thread can use signals."""
    if threading.current_thread() is not threading.main_thread() or not hasattr(
        signal, "setitimer"
    ):
        yield
        return

    def _expired(signum: int, frame: Any) -> None:
        raise _TooSlow

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _call(requests: list[dict[str, Any]], limit: float) -> tuple[Any, str | None]:
    """Return (summary, None), or (None, the mistake). Never raises, so each test has one site."""
    if _HUNG:
        return None, _HUNG[0]
    try:
        with _time_box(limit):
            return latency_summary(requests), None
    except _TooSlow:
        _HUNG.append(SLOW_MESSAGE)
        return None, SLOW_MESSAGE
    except Exception as exc:
        return None, (
            f"your function raised {type(exc).__name__} on valid requests: check the token list "
            "handling and the percentile positions"
        )


def _exact_percentile(values: list[Fraction], p: int) -> Fraction:
    """Linear interpolation done in exact fractions, so the oracle has no rounding."""
    ordered = sorted(values)
    m = len(ordered)
    h = Fraction(m - 1) * p / 100
    lo = h.numerator // h.denominator
    if lo + 1 >= m:
        return ordered[m - 1]
    return ordered[lo] + (h - lo) * (ordered[lo + 1] - ordered[lo])


def _oracle(requests: list[dict[str, Any]]) -> dict[str, float | None]:
    ttft = []
    e2e = []
    itl = []
    for request in requests:
        arrival = Fraction(request["arrival"])
        times = [Fraction(t) for t in request["token_times"]]
        ttft.append(times[0] - arrival)
        e2e.append(times[-1] - arrival)
        if len(times) >= 2:
            itl.append((times[-1] - times[0]) / (len(times) - 1))
    expected: dict[str, float | None] = {}
    for name, values in (("ttft", ttft), ("e2e", e2e)):
        expected[f"{name}_p50"] = float(_exact_percentile(values, 50))
        expected[f"{name}_p95"] = float(_exact_percentile(values, 95))
    if itl:
        expected["itl_p50"] = float(_exact_percentile(itl, 50))
        expected["itl_p95"] = float(_exact_percentile(itl, 95))
    else:
        expected["itl_p50"] = None
        expected["itl_p95"] = None
    return expected


def _compare(requests: list[dict[str, Any]], limit: float = SMALL_CALL_S) -> None:
    summary, problem = _call(requests, limit)
    if problem is not None:
        pytest.fail(problem)
    if not isinstance(summary, dict) or set(summary) != set(KEYS):
        pytest.fail("return a dict with exactly the six named keys")
    expected = _oracle(requests)
    for key in KEYS:
        want = expected[key]
        got = summary[key]
        if want is None:
            if got is not None:
                pytest.fail(f"{key} must be None when no request has two or more tokens")
            continue
        if type(got) is not float:
            pytest.fail(f"{key} must be a Python float, got {type(got).__name__}")
        if not math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9):
            pytest.fail(_mistake_for(key))


def _mistake_for(key: str) -> str:
    if key.startswith("ttft"):
        return (
            "a TTFT percentile is wrong: TTFT is the first token minus arrival, over every request"
        )
    if key.startswith("itl"):
        return (
            "an ITL percentile is wrong: average each request's n - 1 gaps, then take the "
            "percentile (not the mean) over requests with two or more tokens"
        )
    return (
        "an end-to-end percentile is wrong: count from arrival to the last token, for every request"
    )


@functools.cache
def _invoices() -> tuple[tuple[float, int], ...]:
    """(arrival, size) of every real invoice of the Online Retail II sample.

    The arrival is the time since the first invoice, in hours, and stands for the arrival
    time of a request in seconds: invoices come in bursts, several share a timestamp, and
    quiet stretches separate them. The size is the number of lines of the invoice, capped at
    20, and stands for the number of tokens of the reply. Token timings within a request
    come from `_waits`.
    """
    table = retail_invoices()
    hours = (table["invoice_ts"] - table["invoice_ts"].min()).dt.total_seconds() / 3600.0
    sizes = table["n_lines"].clip(upper=20)
    return tuple(zip(hours.tolist(), sizes.tolist(), strict=True))


@functools.cache
def _waits() -> tuple[float, ...]:
    """Real waiting times: the minutes between consecutive invoices, zero waits left out.

    They stand for the time to the first token and for the gaps between tokens, rescaled to
    seconds by the caller. No dataset here records per-token times, so these are a stand-in;
    what they bring is the shape of real waits: most are short, a few are very long.
    """
    stamps = retail_invoices()["invoice_ts"].sort_values()
    minutes = stamps.diff().dt.total_seconds().dropna() / 60.0
    return tuple(minutes[minutes > 0].tolist())


def _random_requests(rng: random.Random, count: int, base: float = 0.0) -> list[dict[str, Any]]:
    """Real arrivals and sizes; first-token delay and token gaps are real waits over 1,000."""
    requests = []
    for _ in range(count):
        hours, n = rng.choice(_invoices())
        arrival = base + hours
        gaps = [rng.choice(_waits()) / 1000.0 for _ in range(n)]
        first = arrival + rng.choice(_waits()) / 1000.0
        times = [first]
        for gap in gaps[1:]:
            times.append(times[-1] + gap)
        requests.append({"arrival": arrival, "token_times": times})
    return requests


def test_seeded_random_requests_match_exact_fraction_oracle() -> None:
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic requests, not security.
    for _ in range(200):
        _compare(_random_requests(rng, rng.randint(1, 40)))


def test_single_request_gives_every_percentile_its_only_value() -> None:
    _compare([{"arrival": 2.0, "token_times": [2.5, 2.75, 3.25]}])


def test_all_one_token_requests_leave_itl_undefined() -> None:
    requests = [{"arrival": float(i), "token_times": [i + 0.1 * i]} for i in range(6)]
    summary, problem = _call(requests, SMALL_CALL_S)
    if problem is not None:
        pytest.fail(problem)
    if summary["itl_p50"] is not None or summary["itl_p95"] is not None:
        pytest.fail("with no request of two or more tokens, both ITL values must be None")
    _compare(requests)


def test_itl_is_averaged_per_request_not_over_pooled_gaps() -> None:
    slow_stream = {"arrival": 0.0, "token_times": [0.0, 1.0]}
    fast_stream = {"arrival": 0.0, "token_times": [0.0] + [0.01 * k for k in range(1, 101)]}
    _compare([slow_stream, fast_stream])


def test_end_to_end_counts_from_arrival_not_from_first_token() -> None:
    requests = [
        {"arrival": 0.0, "token_times": [4.0, 4.5]},
        {"arrival": 1.0, "token_times": [1.5, 2.0, 2.5]},
    ]
    _compare(requests)


def test_percentile_uses_linear_interpolation_not_nearest_rank() -> None:
    requests = [{"arrival": 0.0, "token_times": [v]} for v in (1.0, 2.0, 3.0, 4.0)]
    summary, problem = _call(requests, SMALL_CALL_S)
    if problem is not None:
        pytest.fail(problem)
    if abs(summary["ttft_p50"] - 2.5) > 1e-9 or abs(summary["ttft_p95"] - 3.85) > 1e-9:
        pytest.fail(
            "a TTFT percentile is wrong on four values: interpolate between neighbours, "
            "do not round to the nearest rank"
        )


def test_tied_latencies_return_the_tied_value() -> None:
    requests = [{"arrival": 0.0, "token_times": [0.5, 0.75]} for _ in range(7)]
    _compare(requests)


def test_large_timestamps_with_small_gaps_keep_their_precision() -> None:
    _compare(_random_requests(random.Random(5), 30, base=1_000_000.0))  # noqa: S311


def test_integer_and_float_times_mix_freely() -> None:
    requests = [
        {"arrival": 0, "token_times": [1, 2, 4]},
        {"arrival": 1.5, "token_times": [2.0, 3]},
        {"arrival": 2, "token_times": [3.0]},
    ]
    _compare(requests)


def test_large_request_list_finishes_within_the_time_limit() -> None:
    requests = _random_requests(random.Random(11), LARGE_N)  # noqa: S311
    summary, problem = _call(requests, LARGE_CALL_S)
    if problem is not None:
        pytest.fail(problem)
    expected = _oracle(requests)
    for key in KEYS:
        want = expected[key]
        if want is None:
            if summary[key] is not None:
                pytest.fail(f"{key} must be None when no request has two or more tokens")
            continue
        if type(summary[key]) is not float or not math.isclose(
            summary[key], want, rel_tol=1e-9, abs_tol=1e-9
        ):
            pytest.fail(_mistake_for(key))


def test_input_requests_are_not_mutated() -> None:
    requests = [
        {"arrival": 0.1, "token_times": [0.3]},
        {"arrival": 0.0, "token_times": [0.2, 0.4, 0.9]},
    ]
    before = copy.deepcopy(requests)
    _, problem = _call(requests, SMALL_CALL_S)
    if problem is not None:
        pytest.fail(problem)
    if requests != before:
        pytest.fail("the input requests were changed: sort copies of the lists, not the inputs")


def _expect(exc_type: type[Exception], call: Any, mistake: str) -> None:
    try:
        call()
    except exc_type:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; instead it raised {type(exc).__name__}")
    pytest.fail(mistake)


@pytest.mark.parametrize(
    "requests",
    [
        [{"arrival": 0.0, "token_times": [True]}],
        [{"arrival": "0", "token_times": [1.0]}],
        [{"arrival": 0.0, "token_times": [None]}],
        [{"arrival": 0.0, "token_times": (1.0, 2.0)}],
        ["not a request"],
        [{"arrival": None, "token_times": [1.0]}],
    ],
    ids=[
        "bool-time",
        "string-arrival",
        "none-token",
        "tuple-token-times",
        "not-a-dict",
        "none-arrival",
    ],
)
def test_wrong_types_raise_type_error(requests: list[Any]) -> None:
    _expect(
        TypeError,
        lambda: latency_summary(requests),
        "a value of the wrong type must raise TypeError, not be coerced",
    )


@pytest.mark.parametrize(
    "requests",
    [
        [],
        [{"arrival": 0.0}],
        [{"token_times": [1.0]}],
        [{"arrival": 0.0, "token_times": []}],
        [{"arrival": 0.0, "token_times": [1.0, 1.0]}],
        [{"arrival": 0.0, "token_times": [2.0, 1.0]}],
        [{"arrival": 5.0, "token_times": [1.0]}],
        [{"arrival": 0.0, "token_times": [math.inf]}],
        [{"arrival": math.nan, "token_times": [1.0]}],
    ],
    ids=[
        "empty-list",
        "missing-token-times",
        "missing-arrival",
        "no-tokens",
        "repeated-token-time",
        "decreasing-token-times",
        "first-token-before-arrival",
        "infinite-time",
        "nan-arrival",
    ],
)
def test_invalid_request_values_raise_value_error(requests: list[Any]) -> None:
    _expect(
        ValueError,
        lambda: latency_summary(requests),
        "a missing key, an empty token list, non-increasing times or a non-finite time "
        "must raise ValueError",
    )


def test_first_token_before_arrival_is_rejected_when_later_tokens_are_after_arrival() -> None:
    # The first token must not appear before the request arrives, even if later tokens do.
    _expect(
        ValueError,
        lambda: latency_summary([{"arrival": 1.0, "token_times": [0.5, 2.0]}]),
        "a first token that appears before the request arrives must raise ValueError, "
        "even when later tokens come after arrival",
    )
