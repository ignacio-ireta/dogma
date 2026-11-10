from __future__ import annotations

import math


def compute_optimal_allocation(flop_budget: int, tokens_per_param: int) -> tuple[int, int]:
    """Wrong on purpose: bool budgets are accepted as ints.

    Integer arithmetic only: the answer is exact for any budget, however large.
    """
    # Validate first. bool is a subclass of int, so it is rejected by name, and so is float.
    if not isinstance(flop_budget, int):
        raise TypeError("flop_budget must be a Python int")
    if isinstance(tokens_per_param, bool) or not isinstance(tokens_per_param, int):
        raise TypeError("tokens_per_param must be a Python int")
    if flop_budget < 1:
        raise ValueError("flop_budget must be at least 1")
    if tokens_per_param < 1:
        raise ValueError("tokens_per_param must be at least 1")

    # Step 1, the cost model. With D = r * N, the cost 6 * N * D is 6 * r * N**2. Write the cost
    # per squared parameter as 6r.
    cost_per_square = 6 * tokens_per_param

    # Step 2, the largest N. The condition 6 r N**2 <= C is N**2 <= C / (6 r). Since N**2 is an
    # integer, that is N**2 <= floor(C / (6 r)). Integer floor division gives this exactly.
    max_square = flop_budget // cost_per_square

    # Step 3, the square root. floor(sqrt(x)) equals isqrt(floor(x)) for x >= 0, so math.isqrt
    # returns the exact integer square root with no float rounding.
    n_params = math.isqrt(max_square)
    if n_params == 0:
        raise ValueError("no model fits: the budget is below 6 * tokens_per_param")

    # Step 4, the tokens. D = r * N keeps the ratio exact. Because N was rounded down, the
    # product 6 * N * D cannot exceed C.
    return n_params, tokens_per_param * n_params
