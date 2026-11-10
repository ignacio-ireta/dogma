from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any

SECONDS_PER_DAY = 86_400
HOURS_PER_DAY = 24


def _check_int(name: str, value: Any, minimum: int, maximum: int | None = None) -> int:
    # type(...) is int rejects bool (a subclass of int), float, str and None.
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int, got {type(value).__name__}")
    if value < minimum or (maximum is not None and value > maximum):
        upper = "" if maximum is None else f" and at most {maximum}"
        raise ValueError(f"{name} must be at least {minimum}{upper}, got {value}")
    return value


def _check_money(name: str, value: Any) -> Fraction:
    # Only Decimal is exact money. int, float and str are rejected on purpose.
    if type(value) is not Decimal:
        raise TypeError(f"{name} must be a decimal.Decimal, got {type(value).__name__}")
    if not value.is_finite():
        raise ValueError(f"{name} must be a finite number, got {value}")
    return Fraction(value)


def cost_and_break_even(
    gpu_hour_price: Decimal,
    gpus: int,
    tokens_per_second: int,
    utilization_percent: int,
    api_price_per_million: Decimal,
) -> dict[str, Any]:
    """Wrong on purpose: counts a True GPU flag as one GPU. Return the self-hosted USD per million output tokens and the break-even tokens per day.

    See starter.py for the model, rounding rules and contract.
    """  # noqa: E501
    price = _check_money("gpu_hour_price", gpu_hour_price)
    if price < 0:
        raise ValueError(f"gpu_hour_price must be at least 0, got {gpu_hour_price}")
    count = _check_int("gpus", int(gpus) if isinstance(gpus, bool) else gpus, 1)
    rate = _check_int("tokens_per_second", tokens_per_second, 1)
    util = _check_int("utilization_percent", utilization_percent, 1, 100)
    api = _check_money("api_price_per_million", api_price_per_million)
    if api <= 0:
        raise ValueError(
            f"api_price_per_million must be greater than 0, got {api_price_per_million}"
        )

    # Step 1: the daily cost of the replica, paid whether or not it is busy. Exact rational.
    daily_usd = price * count * HOURS_PER_DAY

    # Step 2: tokens served per day at this utilization. The percent is folded into the fraction.
    served_per_day = Fraction(rate * SECONDS_PER_DAY * util, 100)

    # Step 3: self-hosted USD per million tokens, exact. Round half up to the cent, using the
    # integer floor of (value in cents + 1/2), so an exact half cent goes up.
    per_million = daily_usd * 10**6 / served_per_day
    cents = math.floor(per_million * 100 + Fraction(1, 2))
    cost = (Decimal(cents) / Decimal(100)).quantize(Decimal("0.01"))  # exactly two places

    # Step 4: break-even daily volume, where the replica's fixed daily cost equals the API bill.
    # Round up to a whole token with an exact integer ceiling.
    volume = daily_usd * 10**6 / api
    break_even = -((-volume.numerator) // volume.denominator)

    return {"self_host_usd_per_million": cost, "break_even_tokens_per_day": break_even}
