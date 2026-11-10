from __future__ import annotations

from typing import Any


class LRUCache:
    """A fixed-capacity cache that evicts the least recently used entry.

    Each node is a four-item list [key, value, prev, next] instead of an object.
    The recency list runs from the head sentinel (most recent) to the tail sentinel (least recent).
    """

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a Python int of at least 1")
        self._capacity = capacity
        self._nodes: dict[Any, list[Any]] = {}
        self._head: list[Any] = [None, None, None, None]
        self._tail: list[Any] = [None, None, self._head, None]
        self._head[3] = self._tail

    def _unlink(self, node: list[Any]) -> None:
        prev, nxt = node[2], node[3]
        prev[3] = nxt
        nxt[2] = prev

    def _push_front(self, node: list[Any]) -> None:
        first = self._head[3]
        node[2] = self._head
        node[3] = first
        first[2] = node
        self._head[3] = node

    def get(self, key: Any) -> Any | None:
        node = self._nodes.get(key)
        if node is None:
            return None
        self._unlink(node)
        self._push_front(node)
        return node[1]

    def put(self, key: Any, value: Any) -> None:
        if value is None:
            raise ValueError("None cannot be stored: get returns None for a miss")
        node = self._nodes.get(key)
        if node is not None:
            node[1] = value
            self._unlink(node)
            self._push_front(node)
            return
        if len(self._nodes) == self._capacity:
            victim = self._tail[2]
            self._unlink(victim)
            del self._nodes[victim[0]]
        node = [key, value, None, None]
        self._nodes[key] = node
        self._push_front(node)

    def __len__(self) -> int:
        return len(self._nodes)
