from __future__ import annotations

from collections.abc import Sequence


def count_dense_parameters(widths: Sequence[int], use_bias: bool = True) -> tuple[list[int], int]:
    """Return per-layer and total parameter counts for a dense chain.

    widths is a list/tuple of at least two positive Python integers (not bool).
    use_bias is bool. Invalid input raises ValueError. No mutation or allocation
    of model arrays. Counts are exact Python integers."""
    raise NotImplementedError
