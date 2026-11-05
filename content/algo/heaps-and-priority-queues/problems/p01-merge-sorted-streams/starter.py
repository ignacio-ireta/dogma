from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any


def merge_sorted_streams(streams: Sequence[Iterable[tuple[Any, Any]]]) -> Iterator[tuple[Any, Any]]:
    """Lazily merge k sorted streams of (key, label) pairs into one sorted iterator.

    streams: k iterables, each yielding pairs with non-decreasing keys. Labels are never compared.
    Calling this function reads nothing. The output is sorted by key; among equal keys, a lower
    stream index comes first, and pairs from one stream keep their order. Each stream is read at
    most one pair beyond what the caller has taken from it. Runs in O(n log k) with a heap.

    Returns
    -------
    An iterator of (key, label) pairs.
    """
    raise NotImplementedError
