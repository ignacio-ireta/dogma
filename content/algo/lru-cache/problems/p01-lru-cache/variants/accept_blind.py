from __future__ import annotations

from typing import Any


class _Node:
    __slots__ = ("key", "next", "prev", "value")

    def __init__(self, key: Any = None, value: Any = None) -> None:
        self.key = key
        self.value = value
        self.prev: _Node | None = None
        self.next: _Node | None = None


class LRUCache:
    """A fixed-capacity cache that evicts the least recently used entry.

    Contract
    --------
    capacity : Python int, at least 1. Anything else raises ValueError (bool, float and None too).
    get(key) : returns the stored value and makes key the most recent entry. A miss returns None
        and changes nothing.
    put(key, value) : value must not be None (raises ValueError). An existing key is overwritten
        and made most recent, with no eviction. A new key evicts the least recent entry when the
        cache is full, and is then stored as the most recent entry.
    len(cache) : the number of stored entries, never more than capacity.

    Both get and put run in O(1) expected time. Do not use collections or functools.
    """

    def __init__(self, capacity: int) -> None:
        """Create an empty cache. Raise ValueError unless capacity is a Python int >= 1."""
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a Python int of at least 1")
        self._capacity = capacity
        self._map: dict[Any, _Node] = {}
        # Sentinels: head.next is the most recent entry, tail.prev the least recent.
        self._head = _Node()
        self._tail = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head

    def _unlink(self, node: _Node) -> None:
        assert node.prev is not None and node.next is not None
        node.prev.next = node.next
        node.next.prev = node.prev

    def _push_front(self, node: _Node) -> None:
        first = self._head.next
        assert first is not None
        node.prev = self._head
        node.next = first
        self._head.next = node
        first.prev = node

    def get(self, key: Any) -> Any | None:
        """Return the value stored under key and mark it most recent, or None on a miss."""
        node = self._map.get(key)
        if node is None:
            return None
        self._unlink(node)
        self._push_front(node)
        return node.value

    def put(self, key: Any, value: Any) -> None:
        """Store value under key as the most recent entry, evicting the least recent if full.

        Raise ValueError when value is None.
        """
        if value is None:
            raise ValueError("None cannot be stored; it is the miss result of get")
        node = self._map.get(key)
        if node is not None:
            node.value = value
            self._unlink(node)
            self._push_front(node)
            return
        if len(self._map) >= self._capacity:
            victim = self._tail.prev
            assert victim is not None and victim is not self._head
            self._unlink(victim)
            del self._map[victim.key]
        node = _Node(key, value)
        self._map[key] = node
        self._push_front(node)

    def __len__(self) -> int:
        """Return the number of stored entries."""
        return len(self._map)
