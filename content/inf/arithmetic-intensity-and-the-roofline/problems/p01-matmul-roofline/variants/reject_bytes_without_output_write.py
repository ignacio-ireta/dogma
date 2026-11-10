from __future__ import annotations

from typing import Any


def _check_int(name: str, value: Any) -> int:
    # type(...) is int rejects bool (a subclass of int), float, str and None.
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def matmul_roofline(  # noqa: PLR0917 (the contract fixes six arguments)
    m: int,
    n: int,
    k: int,
    element_bytes: int,
    peak_flops: int,
    bandwidth: int,
) -> dict[str, Any]:
    """Wrong on purpose: the work, traffic, roofline bound and time floor of one product C = A B.

    See starter.py for the full contract.
    """
    m = _check_int("m", m)
    n = _check_int("n", n)
    k = _check_int("k", k)
    element_bytes = _check_int("element_bytes", element_bytes)
    peak_flops = _check_int("peak_flops", peak_flops)
    bandwidth = _check_int("bandwidth", bandwidth)

    # Step 1: work and traffic as exact integers. Each output is a dot product of length k:
    # k multiplies and k adds, so 2 * k FLOPs per output and 2 * m * n * k in total.
    flops = 2 * m * n * k
    # Step 2: bytes moved, each element of A, B and C counted once (the ideal, no-reuse count).
    traffic = (m * k + k * n) * element_bytes

    # Step 3: arithmetic intensity I = F / Q, in FLOP per byte. A float is fine for reporting.
    intensity = flops / traffic

    # Step 4: which roof binds. Intensity >= ridge (peak / bandwidth) means compute-bound. Multiply
    # both sides out to integers so the comparison is exact. A tie at the ridge counts as compute.
    compute_bound = flops * bandwidth >= peak_flops * traffic

    # Step 5: the attainable rate is the lower of the two roofs, min(peak, intensity * bandwidth).
    attainable = float(peak_flops) if compute_bound else intensity * bandwidth

    # Step 6: the time floor is the larger of the compute time and the memory time.
    time_s = max(flops / peak_flops, traffic / bandwidth)

    return {
        "flops": flops,
        "bytes": traffic,
        "intensity": intensity,
        "attainable_flops": attainable,
        "bound": "compute" if compute_bound else "memory",
        "time_s": time_s,
    }
