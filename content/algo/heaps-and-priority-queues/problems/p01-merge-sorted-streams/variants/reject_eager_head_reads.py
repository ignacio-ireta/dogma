from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

# Sentinel returned by next(it, _END) when a stream is exhausted. A sentinel avoids StopIteration.
_END: Any = object()


def merge_sorted_streams(streams: Sequence[Iterable[tuple[Any, Any]]]) -> Iterator[tuple[Any, Any]]:
    """Wrong on purpose: reads the first pair of every stream at call time. Lazily merge k sorted streams.

    Returns
    -------
    An iterator of (key, label) pairs.
    """  # noqa: E501
    iterators = [iter(stream) for stream in streams]
    heap: list[tuple[Any, int, Any]] = []
    for index, iterator in enumerate(iterators):
        first = next(iterator, _END)
        if first is not _END:
            key, label = first
            heapq.heappush(heap, (key, index, label))
    return _drain(heap, iterators)


def _drain(heap: list[tuple[Any, int, Any]], iterators: list[Any]) -> Iterator[tuple[Any, Any]]:
    while heap:
        key, index, label = heapq.heappop(heap)
        yield key, label
        following = next(iterators[index], _END)
        if following is not _END:
            next_key, next_label = following
            heapq.heappush(heap, (next_key, index, next_label))
