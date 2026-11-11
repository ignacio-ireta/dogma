from __future__ import annotations

import itertools
import math
from typing import Any


def _number(name: str, value: Any) -> float:
    # type(...) is not bool rejects bool, None and strings before any arithmetic.
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an int or float, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _request_times(request: Any) -> tuple[float, list[float]]:
    """Validate one request and return (arrival, token_times as floats)."""
    if type(request) is not dict:
        raise TypeError(f"each request must be a dict, got {type(request).__name__}")
    for key in ("arrival", "token_times"):
        if key not in request:
            raise ValueError(f"request is missing the key {key}")
    arrival = _number("arrival", request["arrival"])
    if type(request["token_times"]) is not list:
        raise TypeError("token_times must be a list")
    if not request["token_times"]:
        raise ValueError("token_times must hold at least one token time")
    times = [_number("token time", t) for t in request["token_times"]]
    if times[0] < arrival:
        raise ValueError("the first token cannot appear before the request arrives")
    if any(later <= earlier for earlier, later in itertools.pairwise(times)):
        raise ValueError("token_times must be strictly increasing")
    return arrival, times


def serving_rates(
    requests: list[dict[str, Any]],
    ttft_slo: float,
    itl_slo: float,
) -> tuple[float, float]:
    """Wrong on purpose: a string SLO bound is converted to a number instead of raising TypeError. Return (throughput in tokens per second, goodput in requests per second).

    See starter.py for the full contract.
    """  # noqa: E501
    if type(requests) is not list:
        raise TypeError(f"requests must be a list, got {type(requests).__name__}")
    if not requests:
        raise ValueError("at least one request is required")
    ttft_bound = float(ttft_slo)
    itl_bound = _number("itl_slo", itl_slo)
    if ttft_bound < 0 or itl_bound < 0:
        raise ValueError("the SLO bounds must be zero or positive")

    first_arrival = math.inf
    last_token = -math.inf
    total_tokens = 0
    meeting = 0
    for request in requests:
        arrival, times = _request_times(request)
        # Step 1: the window spans from the earliest arrival to the latest token of any request.
        first_arrival = min(first_arrival, arrival)
        last_token = max(last_token, times[-1])
        # Step 2: throughput counts every token, including the first token of each request.
        total_tokens += len(times)
        # Step 3: a request meets the SLO when TTFT is within bound, and its mean ITL is within
        # bound. A one-token request has no gaps, so only the TTFT bound applies to it.
        ttft = times[0] - arrival
        if len(times) >= 2:
            mean_itl = (times[-1] - times[0]) / (len(times) - 1)
            itl_ok = mean_itl <= itl_bound
        else:
            itl_ok = True
        if ttft <= ttft_bound and itl_ok:
            meeting += 1

    window = last_token - first_arrival
    if window <= 0:
        raise ValueError("the measurement window has no duration")
    return total_tokens / window, meeting / window
