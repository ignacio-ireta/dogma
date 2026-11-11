from __future__ import annotations

import math
from typing import Any


def _time(name: str, value: Any) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an int or float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _percentile(sorted_values: list[float], p: int) -> float:
    m = len(sorted_values)
    h = (m - 1) * p / 100
    lo = math.floor(h)
    if lo + 1 >= m:
        return sorted_values[m - 1]
    return sorted_values[lo] + (h - lo) * (sorted_values[lo + 1] - sorted_values[lo])


def latency_summary(requests: list[dict[str, Any]]) -> dict[str, float | None]:
    if not isinstance(requests, list) or not requests:
        raise ValueError("at least one request is required")
    ttft, e2e, itl = [], [], []
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
        ttft.append(times[0] - arrival)
        e2e.append(times[-1] - arrival)
        if len(times) >= 2:
            itl.append((times[-1] - times[0]) / (len(times) - 1))
    ttft.sort()
    e2e.sort()
    itl.sort()
    out: dict[str, float | None] = {
        "ttft_p50": _percentile(ttft, 50),
        "ttft_p95": _percentile(ttft, 95),
        "e2e_p50": _percentile(e2e, 50),
        "e2e_p95": _percentile(e2e, 95),
        "itl_p50": None,
        "itl_p95": None,
    }
    if itl:
        out["itl_p50"] = _percentile(itl, 50)
        out["itl_p95"] = _percentile(itl, 95)
    return out
