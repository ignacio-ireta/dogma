from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise


def count_dense_parameters(widths: Sequence[int], use_bias: bool = True) -> tuple[list[int], int]:
    """Wrong on purpose: returns per-layer and total parameter counts.

    widths is a list/tuple of at least two positive Python integers (not bool).
    use_bias is bool. Invalid inputs raise ValueError. No mutation or matrices.
    """
    if not isinstance(widths, (list, tuple)):
        raise ValueError("widths must be a list or tuple with at least two entries")
    if any(type(width) is not int or width <= 0 for width in widths):
        raise ValueError("widths must contain positive Python integers")
    if type(use_bias) is not bool:
        raise ValueError("use_bias must be bool")
    # A matrix contributes a*b independent slots; its output bias contributes b.
    per_layer = [a * b + (b if use_bias else 0) for a, b in pairwise(widths)]
    return per_layer, sum(per_layer)
