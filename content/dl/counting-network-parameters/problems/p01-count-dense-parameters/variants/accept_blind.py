from __future__ import annotations

from collections.abc import Sequence


def count_dense_parameters(widths: Sequence[int], use_bias: bool = True) -> tuple[list[int], int]:
    """Return per-layer and total parameter counts for a dense chain.

    widths is a list/tuple of at least two positive Python integers (not bool).
    use_bias is bool. Invalid input raises ValueError. No mutation or allocation
    of model arrays. Counts are exact Python integers."""
    if not isinstance(widths, (list, tuple)):
        raise ValueError("widths must be a list or tuple")
    if len(widths) < 2:
        raise ValueError("widths must have at least two entries")
    for w in widths:
        if isinstance(w, bool) or not isinstance(w, int) or w <= 0:
            raise ValueError("widths must be positive Python integers")
    if not isinstance(use_bias, bool):
        raise ValueError("use_bias must be a bool")

    per_layer: list[int] = []
    for a, b in zip(widths[:-1], widths[1:]):  # noqa: B905, RUF007
        count = a * b + (b if use_bias else 0)
        per_layer.append(int(count))
    return per_layer, int(sum(per_layer))
