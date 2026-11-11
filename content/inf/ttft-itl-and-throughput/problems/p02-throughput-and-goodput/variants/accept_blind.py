from __future__ import annotations

import math
from typing import Any


def _time(name: str, value: Any) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an int or float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _slo(name: str, value: Any) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an int or float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return float(value)


def serving_rates(
    requests: list[dict[str, Any]], ttft_slo: float, itl_slo: float
) -> tuple[float, float]:
    if not isinstance(requests, list) or not requests:
        raise ValueError("at least one request is required")
    ttft_bound = _slo("ttft_slo", ttft_slo)
    itl_bound = _slo("itl_slo", itl_slo)
    total_tokens = 0
    meeting = 0
    first_arrival = None
    last_token = None
    for req in requests:
        if not isinstance(req, dict):
            raise TypeError("each request must be a dict")
        if "arrival" not in req or "token_times" not in req:
            raise ValueError("each request needs arrival and token_times")
        arrival = _time("arrival", req["arrival"])
        times_raw = req["token_times"]
        if not isinstance(times_raw, list):
            raise TypeError("token_times must be a list")
        if not times_raw:
            raise ValueError("token_times must be non-empty")
        times = [_time("token time", t) for t in times_raw]
        for prev, cur in zip(times, times[1:]):  # noqa: B905, RUF007
            if not cur > prev:
                raise ValueError("token_times must be strictly increasing")
        if arrival > times[0]:
            raise ValueError("the first token cannot come before arrival")
        total_tokens += len(times)
        first_arrival = arrival if first_arrival is None else min(first_arrival, arrival)
        last_token = times[-1] if last_token is None else max(last_token, times[-1])
        ttft = times[0] - arrival
        n = len(times)
        ok_ttft = ttft <= ttft_bound
        ok_itl = True if n == 1 else (times[-1] - times[0]) / (n - 1) <= itl_bound
        if ok_ttft and ok_itl:
            meeting += 1
    window = last_token - first_arrival
    if window <= 0:
        raise ValueError("the measurement window has no duration")
    return float(total_tokens / window), float(meeting / window)
