"""Hidden tests: an independent Decimal-based computation over seeded random prices, shapes and
utilizations, exact half-cent rounding, the ceiling of the break-even volume, utilization
properties, very large values, the return types, and validation.

Failure messages name the kind of mistake. They never print hidden inputs or expected values.
"""

from __future__ import annotations

import math
import random
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction
from typing import Any

import pytest
from submission import cost_and_break_even


def _oracle(price: Decimal, gpus: int, rate: int, util: int, api: Decimal) -> dict[str, Any]:
    """Independent route: Decimal division at high precision, quantised half up to the cent."""
    with localcontext() as ctx:
        ctx.prec = 80
        daily = price * gpus * 24
        served = Decimal(rate * 86_400 * util) / Decimal(100)
        per_million = daily * Decimal(10) ** 6 / served
        cost = per_million.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    exact_volume = Fraction(daily) * 10**6 / Fraction(api)
    return {"self_host_usd_per_million": cost, "break_even_tokens_per_day": math.ceil(exact_volume)}


def _expect_error(exc_type: type[Exception], call: Any, mistake: str) -> None:
    try:
        call()
    except exc_type:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; instead it raised {type(exc).__name__}")
    pytest.fail(mistake)


def test_seeded_random_inputs_match_independent_decimal_computation() -> None:
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic prices, not security
    for _ in range(300):
        price = Decimal(rng.randint(0, 10_000)) / Decimal(100)
        gpus = rng.randint(1, 16)
        rate = rng.randint(1, 50_000)
        util = rng.randint(1, 100)
        api = Decimal(rng.randint(1, 50_000)) / Decimal(100)
        want = _oracle(price, gpus, rate, util, api)
        got = cost_and_break_even(price, gpus, rate, util, api)
        if got["self_host_usd_per_million"] != want["self_host_usd_per_million"]:
            pytest.fail(
                "the self-hosted cost per million disagrees: it is daily cost over tokens per day, "
                "times a million, rounded half up to the cent"
            )
        if got["break_even_tokens_per_day"] != want["break_even_tokens_per_day"]:
            pytest.fail(
                "the break-even volume disagrees: divide the daily cost by the API price "
                "per token, "
                "then round up to a whole token"
            )


def test_half_cent_rounds_up_not_to_even() -> None:
    got = cost_and_break_even(Decimal("3.00"), 3, 32, 100, Decimal("80.00"))[
        "self_host_usd_per_million"
    ]
    if got != Decimal("78.13"):
        pytest.fail("an exact half cent must round up (half up), so 78.125 becomes 78.13")


def test_half_cent_that_binary_floats_cannot_represent_rounds_up() -> None:
    # The exact cost is 1.005 USD, which a float holds as 1.00499999..., so a float route rounds it
    # down to 1.00. Exact arithmetic gives 1.01 under half up.
    got = cost_and_break_even(Decimal("1809"), 1, 500_000, 100, Decimal("10"))[
        "self_host_usd_per_million"
    ]
    if got != Decimal("1.01"):
        pytest.fail("round the exact cost: a float route rounds a binary-inexact half cent down")


def test_break_even_volume_rounds_up_to_a_whole_token() -> None:
    got = cost_and_break_even(Decimal("2.50"), 1, 1000, 100, Decimal("0.70"))[
        "break_even_tokens_per_day"
    ]
    if got != 85_714_286:
        pytest.fail(
            "the break-even volume is a whole number of tokens, rounded up from 85,714,285.71"
        )


def test_break_even_volume_is_unchanged_by_utilization() -> None:
    low = cost_and_break_even(Decimal("2.50"), 2, 900, 10, Decimal("0.45"))
    high = cost_and_break_even(Decimal("2.50"), 2, 900, 100, Decimal("0.45"))
    if low["break_even_tokens_per_day"] != high["break_even_tokens_per_day"]:
        pytest.fail(
            "the break-even volume must not depend on utilization: utilization sets how much of it "
            "the replica can reach"
        )


def test_free_gpu_costs_nothing_and_breaks_even_at_zero_tokens() -> None:
    got = cost_and_break_even(Decimal("0"), 4, 100, 50, Decimal("0.10"))
    if got["self_host_usd_per_million"] != Decimal("0.00"):
        pytest.fail("a free GPU costs 0.00 per million tokens: the daily cost is zero")
    if got["break_even_tokens_per_day"] != 0:
        pytest.fail("with a zero daily cost the break-even volume is 0 tokens")


def test_cost_falls_as_utilization_rises() -> None:
    previous = None
    for util in range(1, 101):
        cost = cost_and_break_even(Decimal("3.10"), 2, 750, util, Decimal("0.80"))[
            "self_host_usd_per_million"
        ]
        if previous is not None and cost > previous:
            pytest.fail(
                "the cost per million rose as utilization rose: more served tokens lower the cost"
            )
        previous = cost


def test_very_large_throughput_and_price_stay_exact() -> None:
    price = Decimal("1234.5678")
    rate = 10**9
    gpus = 1000
    util = 37
    api = Decimal("9999.99")
    want = _oracle(price, gpus, rate, util, api)
    got = cost_and_break_even(price, gpus, rate, util, api)
    if got != want:
        pytest.fail(
            "very large inputs must stay exact: keep Decimal and Fraction arithmetic, no float"
        )


def test_returns_exact_keys_and_types() -> None:
    got = cost_and_break_even(Decimal("2.50"), 1, 1000, 100, Decimal("0.60"))
    if set(got) != {"self_host_usd_per_million", "break_even_tokens_per_day"}:
        pytest.fail(
            "the result must have exactly the keys self_host_usd_per_million and "
            "break_even_tokens_per_day"
        )
    if type(got["self_host_usd_per_million"]) is not Decimal:
        pytest.fail("the cost per million must be a Decimal, not a float")
    if type(got["break_even_tokens_per_day"]) is not int:
        pytest.fail("the break-even volume must be a Python int number of tokens")
    if got["self_host_usd_per_million"].as_tuple().exponent != -2:
        pytest.fail(
            "the cost per million must have exactly two decimal places, rounded to the cent"
        )


@pytest.mark.parametrize(
    ("price", "gpus", "rate", "util", "api", "bad"),
    [
        (2.5, 1, 1000, 100, Decimal("0.60"), "gpu_hour_price"),
        (2, 1, 1000, 100, Decimal("0.60"), "gpu_hour_price"),
        (Decimal("2.50"), True, 1000, 100, Decimal("0.60"), "gpus"),
        (Decimal("2.50"), 1, 1000.0, 100, Decimal("0.60"), "tokens_per_second"),
        (Decimal("2.50"), 1, 1000, True, Decimal("0.60"), "utilization_percent"),
        (Decimal("2.50"), 1, 1000, 100, 0.6, "api_price_per_million"),
        (Decimal("2.50"), 1, 1000, 100, "0.60", "api_price_per_million"),
    ],
    ids=[
        "float-price",
        "int-price",
        "bool-gpus",
        "float-rate",
        "bool-utilization",
        "float-api",
        "str-api",
    ],
)
def test_wrong_type_raises_type_error(  # noqa: PLR0917 (parametrized arguments)
    price: Any, gpus: Any, rate: Any, util: Any, api: Any, bad: str
) -> None:
    _expect_error(
        TypeError,
        lambda: cost_and_break_even(price, gpus, rate, util, api),
        f"{bad} has the wrong type: money is a Decimal, and integers are Python int without bool",
    )


@pytest.mark.parametrize(
    ("price", "gpus", "rate", "util", "api"),
    [
        (Decimal("NaN"), 1, 1000, 100, Decimal("0.60")),
        (Decimal("Infinity"), 1, 1000, 100, Decimal("0.60")),
        (Decimal("-1.00"), 1, 1000, 100, Decimal("0.60")),
        (Decimal("2.50"), 0, 1000, 100, Decimal("0.60")),
        (Decimal("2.50"), 1, 0, 100, Decimal("0.60")),
        (Decimal("2.50"), 1, 1000, 0, Decimal("0.60")),
        (Decimal("2.50"), 1, 1000, 101, Decimal("0.60")),
        (Decimal("2.50"), 1, 1000, 100, Decimal("0")),
        (Decimal("2.50"), 1, 1000, 100, Decimal("-0.60")),
        (Decimal("2.50"), 1, 1000, 100, Decimal("NaN")),
    ],
    ids=[
        "nan-price",
        "infinite-price",
        "negative-price",
        "zero-gpus",
        "zero-rate",
        "zero-utilization",
        "utilization-over-100",
        "zero-api-price",
        "negative-api-price",
        "nan-api-price",
    ],
)
def test_out_of_range_money_or_count_raises_value_error(
    price: Decimal, gpus: int, rate: int, util: int, api: Decimal
) -> None:
    _expect_error(
        ValueError,
        lambda: cost_and_break_even(price, gpus, rate, util, api),
        "a price that is not finite, negative or zero where it divides, or a count out of range, "
        "must raise ValueError",
    )


def test_negative_gpu_price_just_below_zero_raises_value_error() -> None:
    _expect_error(
        ValueError,
        lambda: cost_and_break_even(Decimal("-0.01"), 1, 1000, 100, Decimal("0.60")),
        "a GPU price below zero must raise ValueError: the price must be at least 0",
    )


def test_tiny_positive_api_price_is_accepted() -> None:
    # Hand worked: daily cost 2.50 * 1 * 24 = 60 USD; 60 * 10**6 / 0.0000005 = 1.2e14 tokens
    try:
        got = cost_and_break_even(Decimal("2.50"), 1, 1000, 100, Decimal("0.0000005"))
    except ValueError:
        pytest.fail("a tiny positive API price is valid: it must not raise ValueError")
    assert got["break_even_tokens_per_day"] == 120_000_000_000_000, (
        "wrong break-even volume for a tiny positive API price: divide by that price"
    )
