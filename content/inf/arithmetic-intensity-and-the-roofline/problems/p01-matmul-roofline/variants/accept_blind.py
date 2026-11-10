from __future__ import annotations

from typing import Any


def matmul_roofline(  # noqa: PLR0917 (the contract fixes six arguments)
    m: int,
    n: int,
    k: int,
    element_bytes: int,
    peak_flops: int,
    bandwidth: int,
) -> dict[str, Any]:
    """Return the work, traffic, roofline bound and time floor of one product C = A B.

    A is m x k and B is k x n. FLOPs are 2 * m * n * k. Bytes are
    (m*k + k*n + m*n) * element_bytes.
    The bound is "compute" when flops * bandwidth >= peak_flops * bytes (a tie is
    compute), else "memory". The attainable rate is peak_flops when compute-bound, else
    intensity * bandwidth. The time floor is max(flops / peak_flops, bytes / bandwidth) in
    seconds. All of m, n, k, element_bytes, peak_flops and bandwidth must be Python ints
    (TypeError otherwise) and at least 1 (ValueError otherwise).

    Returns a dict with keys "flops" (int), "bytes" (int), "intensity" (float), "attainable_flops"
    (float), "bound" ("memory" or "compute") and "time_s" (float). No rounding.
    """
    for name, value in (
        ("m", m),
        ("n", n),
        ("k", k),
        ("element_bytes", element_bytes),
        ("peak_flops", peak_flops),
        ("bandwidth", bandwidth),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a Python int")
    for name, value in (
        ("m", m),
        ("n", n),
        ("k", k),
        ("element_bytes", element_bytes),
        ("peak_flops", peak_flops),
        ("bandwidth", bandwidth),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1")

    flops = 2 * m * n * k
    nbytes = (m * k + k * n + m * n) * element_bytes
    intensity = flops / nbytes
    if flops * bandwidth >= peak_flops * nbytes:
        bound = "compute"
        attainable = float(peak_flops)
    else:
        bound = "memory"
        attainable = intensity * bandwidth
    time_s = max(flops / peak_flops, nbytes / bandwidth)
    return {
        "flops": flops,
        "bytes": nbytes,
        "intensity": intensity,
        "attainable_flops": float(attainable),
        "bound": bound,
        "time_s": float(time_s),
    }
