from __future__ import annotations

from decimal import Decimal
from typing import Any


def cost_and_break_even(
    gpu_hour_price: Decimal,
    gpus: int,
    tokens_per_second: int,
    utilization_percent: int,
    api_price_per_million: Decimal,
) -> dict[str, Any]:
    """Return the self-hosted USD per million output tokens and the break-even tokens per day.

    Daily cost F = gpu_hour_price * gpus * 24 (USD/day). Tokens per day at utilization u is
    tokens_per_second * 86400 * u / 100. Self-hosted USD per million is F * 10**6 / tokens per day,
    rounded to the cent, half up, returned as a Decimal with exactly two places. The break-even
    volume is F * 10**6 / api_price_per_million tokens per day, rounded up to a whole token (int).
    Compute exactly; do not round or convert money to float. TypeError for a wrong type (a Decimal
    price rejects int, float and str; bool is rejected for the integers). ValueError for a price
    that is not finite or is negative, an API price of 0 or less, gpus or tokens_per_second below 1,
    or a utilization outside 1 to 100.
    """
    raise NotImplementedError
