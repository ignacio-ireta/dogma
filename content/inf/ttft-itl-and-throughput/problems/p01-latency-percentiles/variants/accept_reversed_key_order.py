from __future__ import annotations

import itertools
import math
from typing import Any


def _time(name: str, value: Any) -> float:
    # type(...) is not bool and an int/float check rejects bool, None and strings.
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an int or float, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _percentile(sorted_values: list[float], p: float) -> float:
    """Linear interpolation between order statistics (NumPy's default, Hyndman-Fan type 7)."""
    m = len(sorted_values)
    # Position of the percentile among the m sorted values, counted from 0.
    h = (m - 1) * p / 100
    lo = math.floor(h)
    if lo + 1 >= m:
        return sorted_values[m - 1]
    # Interpolate a fraction (h - lo) of the way from x[lo] to x[lo + 1].
    return sorted_values[lo] + (h - lo) * (sorted_values[lo + 1] - sorted_values[lo])


def latency_summary(requests: list[dict[str, Any]]) -> dict[str, float | None]:
    """Return p50 and p95 of TTFT, mean inter-token latency and end-to-end latency.

    See starter.py for the full contract.
    """
    if type(requests) is not list:
        raise TypeError(f"requests must be a list, got {type(requests).__name__}")
    if not requests:
        raise ValueError("at least one request is required")

    ttft: list[float] = []
    e2e: list[float] = []
    itl: list[float] = []
    for request in requests:
        if not isinstance(request, dict):
            raise TypeError(f"each request must be a dict, got {type(request).__name__}")
        for key in ("arrival", "token_times"):
            if key not in request:
                raise ValueError(f"request is missing the key {key}")
        arrival = _time("arrival", request["arrival"])
        times_in = request["token_times"]
        if type(times_in) is not list:
            raise TypeError(f"token_times must be a list, got {type(times_in).__name__}")
        if not times_in:
            raise ValueError("token_times must hold at least one token time")
        times = [_time("token time", t) for t in times_in]
        if times[0] < arrival:
            raise ValueError("the first token cannot appear before the request arrives")
        for earlier, later in itertools.pairwise(times):
            if later <= earlier:
                raise ValueError("token_times must be strictly increasing")

        # Step 1: TTFT is the wait before the first token: queueing plus prefill.
        ttft.append(times[0] - arrival)
        # Step 2: E2E counts from arrival to the last token, so queueing is included.
        e2e.append(times[-1] - arrival)
        # Step 3: mean ITL is the average of the n - 1 gaps, which telescope to last - first.
        # One-token requests have no gaps, so they are left out of the ITL list.
        if len(times) >= 2:
            itl.append((times[-1] - times[0]) / (len(times) - 1))

    ttft.sort()
    e2e.sort()
    itl.sort()
    return {
        "e2e_p95": _percentile(e2e, 95),
        "e2e_p50": _percentile(e2e, 50),
        "itl_p95": _percentile(itl, 95) if itl else None,
        "itl_p50": _percentile(itl, 50) if itl else None,
        "ttft_p95": _percentile(ttft, 95),
        "ttft_p50": _percentile(ttft, 50),
    }
