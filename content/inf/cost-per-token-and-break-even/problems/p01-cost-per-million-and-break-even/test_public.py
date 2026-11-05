"""Public tests: the examples from problem.md, the half-cent tie, and the zero-price edge case."""

from __future__ import annotations

from decimal import Decimal

import pytest
from submission import cost_and_break_even


def test_full_utilization_cost_and_break_even_for_one_gpu() -> None:
    got = cost_and_break_even(Decimal("2.50"), 1, 1000, 100, Decimal("0.60"))
    assert got["self_host_usd_per_million"] == Decimal("0.69"), (
        "wrong cost per million: the daily cost is 60 USD, and the replica serves 86.4e6 "
        "tokens per day at 100 percent, so the cost is 0.694... USD per million, to the cent"
    )
    assert got["break_even_tokens_per_day"] == 100_000_000, (
        "wrong break-even volume: divide the daily cost by the price per token, 60e6 / 0.60"
    )


def test_half_utilization_doubles_the_cost_but_not_the_break_even() -> None:
    got = cost_and_break_even(Decimal("2.50"), 1, 1000, 50, Decimal("0.60"))
    assert got["self_host_usd_per_million"] == Decimal("1.39"), (
        "wrong cost at 50 percent utilization: the same daily cost covers half the tokens"
    )
    assert got["break_even_tokens_per_day"] == 100_000_000, (
        "the break-even volume must not depend on utilization: daily cost over API price"
    )


def test_exact_half_cent_rounds_up() -> None:
    got = cost_and_break_even(Decimal("3.00"), 3, 32, 100, Decimal("80.00"))
    assert got["self_host_usd_per_million"] == Decimal("78.13"), (
        "the exact cost is 78.125, a half cent: round half up to 78.13, not half-even 78.12"
    )


def test_zero_api_price_raises_value_error() -> None:
    try:
        cost_and_break_even(Decimal("2.50"), 1, 1000, 100, Decimal("0"))
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"an API price of 0 must raise ValueError, not {type(exc).__name__}")
    pytest.fail("an API price of 0 must raise ValueError: the break-even volume divides by it")
