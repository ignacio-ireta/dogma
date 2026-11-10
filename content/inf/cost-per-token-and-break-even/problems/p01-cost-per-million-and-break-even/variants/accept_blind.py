from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def cost_and_break_even(
    gpu_hour_price: Decimal,
    gpus: int,
    tokens_per_second: int,
    utilization_percent: int,
    api_price_per_million: Decimal,
) -> dict[str, Any]:
    """Return the self-hosted USD per million output tokens and the break-even tokens per day."""
    if not isinstance(gpu_hour_price, Decimal):
        raise TypeError("gpu_hour_price must be a Decimal")
    if not isinstance(api_price_per_million, Decimal):
        raise TypeError("api_price_per_million must be a Decimal")
    if not _is_int(gpus):
        raise TypeError("gpus must be an int")
    if not _is_int(tokens_per_second):
        raise TypeError("tokens_per_second must be an int")
    if not _is_int(utilization_percent):
        raise TypeError("utilization_percent must be an int")

    if not gpu_hour_price.is_finite() or gpu_hour_price < 0:
        raise ValueError("gpu_hour_price must be finite and at least 0")
    if gpus < 1:
        raise ValueError("gpus must be at least 1")
    if tokens_per_second < 1:
        raise ValueError("tokens_per_second must be at least 1")
    if not 1 <= utilization_percent <= 100:
        raise ValueError("utilization_percent must be between 1 and 100")
    if not api_price_per_million.is_finite() or api_price_per_million <= 0:
        raise ValueError("api_price_per_million must be finite and greater than 0")

    daily_cost = Fraction(gpu_hour_price) * gpus * 24
    tokens_per_day = Fraction(tokens_per_second * 86400 * utilization_percent, 100)
    usd_per_million = daily_cost * 10**6 / tokens_per_day

    # Round to the cent, half up, using exact arithmetic on the hundredths.
    cents = math.floor(usd_per_million * 100 + Fraction(1, 2))
    self_host = Decimal(f"{cents // 100}.{cents % 100:02d}")

    break_even = daily_cost * 10**6 / Fraction(api_price_per_million)
    return {
        "self_host_usd_per_million": self_host,
        "break_even_tokens_per_day": math.ceil(break_even),
    }
