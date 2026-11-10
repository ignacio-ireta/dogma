from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

# Sentinel returned by next(it, _END) when a stream is exhausted. A sentinel avoids StopIteration.
_END: Any = object()


def merge_sorted_streams(streams: Sequence[Iterable[tuple[Any, Any]]]) -> Iterator[tuple[Any, Any]]:
    """Wrong on purpose: reads two pairs ahead of the caller. Lazily merge k sorted streams of (key, label) pairs into one sorted iterator.

    streams: k iterables, each yielding pairs with non-decreasing keys. Labels are never compared.
    Calling this function reads nothing. The output is sorted by key; among equal keys, a lower
    stream index comes first, and pairs from one stream keep their order. Each stream is read at
    most one pair beyond what the caller has taken from it. Runs in O(n log k) with a heap.

    Returns
    -------
    An iterator of (key, label) pairs.
    """  # noqa: E501
    # This is a generator function, so the body runs only on the first next(). iter() reads nothing.
    iterators = [iter(stream) for stream in streams]
    held: dict[int, Any] = {}
    heap: list[tuple[Any, int, Any]] = []
    for index, iterator in enumerate(iterators):
        first = next(iterator, _END)
        if first is not _END:
            held[index] = next(iterator, _END)
            key, label = first
            heapq.heappush(heap, (key, index, label))

    while heap:
        key, index, label = heapq.heappop(heap)
        nxt = held.pop(index)
        if nxt is not _END:
            heapq.heappush(heap, (nxt[0], index, nxt[1]))
            held[index] = next(iterators[index], _END)
        yield key, label
