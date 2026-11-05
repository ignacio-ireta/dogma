from __future__ import annotations

from typing import Any


def latency_summary(requests: list[dict[str, Any]]) -> dict[str, float | None]:
    """Return p50 and p95 of TTFT, mean inter-token latency and end-to-end latency.

    Parameters
    ----------
    requests : non-empty list of dict
        Each dict has "arrival" (int or float, seconds) and "token_times" (non-empty list of int
        or float, strictly increasing, with arrival <= token_times[0]). Not mutated.

    Returns
    -------
    dict
        Keys ttft_p50, ttft_p95, itl_p50, itl_p95, e2e_p50, e2e_p95, each a float. The two itl_
        keys are None when no request has two or more tokens. Percentiles use linear interpolation
        between sorted values. TTFT and E2E use every request; mean ITL uses requests with two or
        more tokens.

    Raises
    ------
    TypeError
        When a request is not a dict, token_times is not a list, or a time is not an int or float.
    ValueError
        When requests is empty, a key is missing, a time is not finite, token_times is empty or not
        strictly increasing, or the first token comes before the arrival.
    """
    raise NotImplementedError
