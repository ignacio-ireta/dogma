from __future__ import annotations

from typing import Any


def serving_rates(
    requests: list[dict[str, Any]],
    ttft_slo: float,
    itl_slo: float,
) -> tuple[float, float]:
    """Return (throughput in tokens per second, goodput in requests per second).

    Parameters
    ----------
    requests : non-empty list of dict
        Each dict has "arrival" and "token_times", as in the latency-percentile problem.
        Not mutated.
    ttft_slo : int or float, finite and >= 0
        A request meets the SLO on TTFT when TTFT <= ttft_slo (inclusive).
    itl_slo : int or float, finite and >= 0
        A request with two or more tokens meets the SLO on mean ITL when mean ITL <= itl_slo
        (inclusive). A one-token request has no ITL and meets this bound.

    Returns
    -------
    tuple of two float
        throughput = total tokens / window, goodput = requests meeting both bounds / window, where
        window = max(last token time) - min(arrival). Every token counts toward throughput.

    Raises
    ------
    TypeError
        When a value is of the wrong type, including bool.
    ValueError
        When a request is malformed, an SLO is negative or non-finite, or the window is
        not positive.
    """
    raise NotImplementedError
