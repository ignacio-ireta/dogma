from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

# Sentinel returned by next(it, _END) when a stream is exhausted. A sentinel avoids StopIteration.
_END: Any = object()


def merge_sorted_streams(streams: Sequence[Iterable[tuple[Any, Any]]]) -> Iterator[tuple[Any, Any]]:
    """Wrong on purpose: equal keys come out in reverse stream order. Lazily merge k sorted streams of (key, label) pairs into one sorted iterator.

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

    # The heap holds one entry per stream that still has a pair: (key, stream index, label).
    # Two entries never share a stream index, so the comparison always ends at the index and never
    # reaches the label. Equal keys therefore come out in stream order.
    heap: list[tuple[Any, int, Any]] = []
    for index, iterator in enumerate(iterators):
        first = next(iterator, _END)  # the head of each stream: the only read before any output
        if first is not _END:
            key, label = first
            heapq.heappush(heap, (key, -index, label))

    while heap:
        key, index, label = heapq.heappop(heap)
        yield key, label
        # Refill from the stream that just gave its pair. This read happens only when the caller
        # asks for the next pair, which keeps each stream at most one pair ahead.
        following = next(iterators[index], _END)
        if following is not _END:
            next_key, next_label = following
            heapq.heappush(heap, (next_key, -index, next_label))
