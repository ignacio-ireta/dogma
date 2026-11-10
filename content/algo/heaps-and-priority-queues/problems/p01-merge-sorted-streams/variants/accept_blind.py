from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator, Sequence
from typing import Any


def merge_sorted_streams(streams: Sequence[Iterable[tuple[Any, Any]]]) -> Iterator[tuple[Any, Any]]:
    """Lazily merge k sorted streams of (key, label) pairs into one sorted iterator."""
    return _merge(streams)


def _merge(streams: Sequence[Iterable[tuple[Any, Any]]]) -> Iterator[tuple[Any, Any]]:
    iters = [iter(s) for s in streams]
    heap: list[tuple[Any, int, Any]] = []
    for index, it in enumerate(iters):
        first = next(it, None)
        if first is not None:
            key, label = first
            heap.append((key, index, label))
    heapq.heapify(heap)
    while heap:
        key, index, label = heapq.heappop(heap)
        yield (key, label)
        nxt = next(iters[index], None)
        if nxt is not None:
            nkey, nlabel = nxt
            heapq.heappush(heap, (nkey, index, nlabel))
