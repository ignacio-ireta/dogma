"""Hidden tests: an exact-fraction oracle on seeded requests, the window and bounds, validation.

Failure messages name the kind of mistake. They never print hidden timestamps or expected values.
"""

from __future__ import annotations

import copy
import math
import random
from fractions import Fraction
from typing import Any

import pytest
from submission import serving_rates


def _pair(got: Any, want: tuple[float, float], hint: str = "") -> None:
    if type(got) is not tuple or len(got) != 2:
        pytest.fail("return a tuple of (throughput, goodput)")
    if type(got[0]) is not float or type(got[1]) is not float:
        pytest.fail("throughput and goodput must be Python floats, not ints or other types")
    if not math.isclose(got[0], want[0], rel_tol=1e-9, abs_tol=1e-9):
        pytest.fail(
            "throughput is wrong: count every token, including first tokens, over a window "
            "that starts at the earliest arrival"
        )
    if not math.isclose(got[1], want[1], rel_tol=1e-9, abs_tol=1e-9):
        pytest.fail(
            hint
            or "goodput is wrong: count requests that meet both bounds, not tokens, over the "
            "same window"
        )


def _oracle(requests: list[dict[str, Any]], ttft_slo: float, itl_slo: float) -> tuple[float, float]:
    """Exact fractions throughout, so the inclusive bounds are decided without rounding."""
    first_arrival = min(Fraction(r["arrival"]) for r in requests)
    last_token = max(Fraction(r["token_times"][-1]) for r in requests)
    window = last_token - first_arrival
    tokens = sum(len(r["token_times"]) for r in requests)
    meeting = 0
    for r in requests:
        arrival = Fraction(r["arrival"])
        times = [Fraction(t) for t in r["token_times"]]
        ttft_ok = times[0] - arrival <= Fraction(ttft_slo)
        if len(times) >= 2:
            itl_ok = (times[-1] - times[0]) / (len(times) - 1) <= Fraction(itl_slo)
        else:
            itl_ok = True
        if ttft_ok and itl_ok:
            meeting += 1
    return float(Fraction(tokens) / window), float(Fraction(meeting) / window)


def _random_requests(rng: random.Random, count: int) -> list[dict[str, Any]]:
    requests = []
    for _ in range(count):
        arrival = rng.uniform(0.0, 20.0)
        n = rng.choice([1, 2, 3, 6, 15])
        times = [arrival + rng.uniform(0.0, 3.0)]
        for _ in range(n - 1):
            times.append(times[-1] + rng.uniform(0.001, 0.5))
        requests.append({"arrival": arrival, "token_times": times})
    return requests


def test_seeded_random_requests_match_exact_fraction_oracle() -> None:
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic requests, not security.
    for _ in range(150):
        requests = _random_requests(rng, rng.randint(1, 30))
        ttft_slo = rng.uniform(0.0, 3.0)
        itl_slo = rng.uniform(0.0, 0.5)
        _pair(serving_rates(requests, ttft_slo, itl_slo), _oracle(requests, ttft_slo, itl_slo))


def test_window_starts_at_the_earliest_arrival_not_the_first_token() -> None:
    requests = [
        {"arrival": 0.0, "token_times": [4.0]},
        {"arrival": 3.0, "token_times": [5.0]},
    ]
    # Window 5.0 s from the earliest arrival: 2 tokens / 5 s, and 2 passing requests / 5 s.
    _pair(serving_rates(requests, 10.0, 10.0), (0.4, 0.4))


def test_every_token_counts_toward_throughput_including_the_first() -> None:
    requests = [{"arrival": 0.0, "token_times": [0.25, 0.5, 0.75, 1.0]}]
    _pair(serving_rates(requests, 1.0, 1.0), (4.0, 1.0))


def test_one_token_requests_are_judged_on_ttft_alone() -> None:
    requests = [{"arrival": 0.0, "token_times": [0.25]}, {"arrival": 0.0, "token_times": [4.0]}]
    _pair(
        serving_rates(requests, 0.5, 0.0),
        (2.0 / 4.0, 1.0 / 4.0),
        hint="a one-token request has no ITL: judge it on TTFT alone",
    )


def test_a_request_with_slow_itl_fails_even_when_ttft_passes() -> None:
    requests = [{"arrival": 0.0, "token_times": [0.25, 0.5, 2.5]}]
    _pair(serving_rates(requests, 1.0, 0.5), (3.0 / 2.5, 0.0))


def test_inclusive_bound_on_itl_is_met_exactly() -> None:
    requests = [{"arrival": 0.0, "token_times": [0.0, 0.5, 1.0]}]
    _pair(
        serving_rates(requests, 0.0, 0.5),
        (3.0 / 1.0, 1.0 / 1.0),
        hint="the ITL bound is inclusive: a mean ITL equal to itl_slo passes",
    )


def test_zero_slo_bounds_are_valid_and_only_exact_hits_pass() -> None:
    requests = [
        {"arrival": 0.0, "token_times": [0.0, 1.0]},
        {"arrival": 0.0, "token_times": [0.25, 1.0]},
    ]
    _pair(serving_rates(requests, 0.0, 1.0), _oracle(requests, 0.0, 1.0))


def test_input_requests_are_not_mutated() -> None:
    requests = [
        {"arrival": 1.0, "token_times": [1.5, 2.0]},
        {"arrival": 0.0, "token_times": [0.25]},
    ]
    before = copy.deepcopy(requests)
    serving_rates(requests, 1.0, 1.0)
    if requests != before:
        pytest.fail("the input requests were changed: read the lists without writing to them")


def _expect(exc_type: type[Exception], call: Any, mistake: str) -> None:
    try:
        call()
    except exc_type:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; instead it raised {type(exc).__name__}")
    pytest.fail(mistake)


@pytest.mark.parametrize(
    ("requests", "ttft_slo", "itl_slo"),
    [
        ([{"arrival": 0.0, "token_times": [1.0]}], True, 1.0),
        ([{"arrival": 0.0, "token_times": [1.0]}], 1.0, "0.5"),
        ([{"arrival": 0.0, "token_times": [1.0]}], None, 1.0),
        ([{"arrival": True, "token_times": [1.0]}], 1.0, 1.0),
        ([{"arrival": 0.0, "token_times": (1.0,)}], 1.0, 1.0),
    ],
    ids=["bool-ttft-slo", "string-itl-slo", "none-ttft-slo", "bool-arrival", "tuple-token-times"],
)
def test_wrong_types_raise_type_error(requests: list[Any], ttft_slo: Any, itl_slo: Any) -> None:
    _expect(
        TypeError,
        lambda: serving_rates(requests, ttft_slo, itl_slo),
        "a value of the wrong type must raise TypeError, not be coerced",
    )


@pytest.mark.parametrize(
    ("requests", "ttft_slo", "itl_slo"),
    [
        ([], 1.0, 1.0),
        ([{"arrival": 0.0, "token_times": [1.0]}], -0.1, 1.0),
        ([{"arrival": 0.0, "token_times": [1.0]}], 1.0, -1.0),
        ([{"arrival": 0.0, "token_times": [1.0]}], math.inf, 1.0),
        ([{"arrival": 0.0, "token_times": [2.0, 1.0]}], 1.0, 1.0),
        ([{"arrival": 2.0, "token_times": [1.0]}], 1.0, 1.0),
        ([{"arrival": 1.0, "token_times": [1.0]}], 0.5, 0.5),
    ],
    ids=[
        "empty-list",
        "negative-ttft-slo",
        "negative-itl-slo",
        "infinite-slo",
        "decreasing-token-times",
        "first-token-before-arrival",
        "zero-length-window",
    ],
)
def test_invalid_values_raise_value_error(
    requests: list[Any], ttft_slo: float, itl_slo: float
) -> None:
    _expect(
        ValueError,
        lambda: serving_rates(requests, ttft_slo, itl_slo),
        "an empty list, a negative or non-finite bound, bad token times or an empty window "
        "must raise ValueError",
    )


def test_request_that_is_not_a_dict_raises_type_error() -> None:
    _expect(
        TypeError,
        lambda: serving_rates(["not a request"], 1.0, 1.0),  # type: ignore[list-item]
        "a request that is not a dict must raise TypeError",
    )


def test_request_missing_arrival_raises_value_error() -> None:
    _expect(
        ValueError,
        lambda: serving_rates([{"token_times": [1.0]}], 1.0, 1.0),
        "a request without an arrival must raise ValueError, not a KeyError",
    )


def test_request_missing_token_times_raises_value_error() -> None:
    _expect(
        ValueError,
        lambda: serving_rates([{"arrival": 0.0}], 1.0, 1.0),
        "a request without token_times must raise ValueError, not a KeyError",
    )


def test_empty_token_list_raises_value_error() -> None:
    _expect(
        ValueError,
        lambda: serving_rates([{"arrival": 0.0, "token_times": []}], 1.0, 1.0),
        "a request with an empty token list must raise ValueError, not an IndexError",
    )


def test_first_token_before_arrival_is_rejected_when_later_tokens_follow_arrival() -> None:
    _expect(
        ValueError,
        lambda: serving_rates([{"arrival": 1.0, "token_times": [0.5, 2.0]}], 1.0, 1.0),
        "a first token that appears before the request arrives must raise ValueError, "
        "even when later tokens come after arrival",
    )
