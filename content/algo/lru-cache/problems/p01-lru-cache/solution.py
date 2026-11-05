from __future__ import annotations

from typing import Any


class _Node:
    """One cache entry, linked into the recency list.

    prev points toward the most recent end (the front), next toward the least recent end (the back).
    """

    __slots__ = ("key", "next", "prev", "value")

    def __init__(self, key: Any, value: Any) -> None:
        self.key = key
        self.value = value
        self.prev: _Node | None = None
        self.next: _Node | None = None


class LRUCache:
    """A fixed-capacity cache that evicts the least recently used entry.

    Two structures, one shared node per entry:
    - ``_nodes`` maps each key to its node, so a lookup is O(1) expected.
    - A doubly linked list, bracketed by ``_head`` and ``_tail`` sentinels, orders the nodes:
      the node after ``_head`` is the most recent, the node before ``_tail`` is the least recent.

    Invariant: ``_nodes`` and the list hold the same entries, and there are at most ``_capacity``.
    """

    def __init__(self, capacity: int) -> None:
        # type(...) is int rejects bool, which is an int subclass, as well as float and None.
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a Python int of at least 1")
        self._capacity = capacity
        self._nodes: dict[Any, _Node] = {}
        # Sentinels hold no data. Every real node always has a neighbour on each side, so
        # unlink and push-front never test for an empty list or the ends.
        self._head = _Node(None, None)
        self._tail = _Node(None, None)
        self._head.next = self._tail
        self._tail.prev = self._head

    def _unlink(self, node: _Node) -> None:
        """Remove node from the list in O(1). The node keeps its key and value."""
        prev, nxt = node.prev, node.next
        if prev is None or nxt is None:
            raise RuntimeError("only a node that is linked into the list can be unlinked")
        prev.next = nxt
        nxt.prev = prev

    def _push_front(self, node: _Node) -> None:
        """Insert node right after the head sentinel, making it the most recent entry."""
        first = self._head.next
        if first is None:
            raise RuntimeError("the head sentinel always has a successor")
        node.prev = self._head
        node.next = first
        first.prev = node
        self._head.next = node

    def get(self, key: Any) -> Any | None:
        node = self._nodes.get(key)
        if node is None:
            # A miss changes nothing: no node moves, so no other entry's recency changes.
            return None
        # A hit is a use. Moving the node to the front protects it from the next eviction.
        self._unlink(node)
        self._push_front(node)
        return node.value

    def put(self, key: Any, value: Any) -> None:
        if value is None:
            raise ValueError("None cannot be stored: get returns None for a miss")
        node = self._nodes.get(key)
        if node is not None:
            # Case 1: existing key. Overwrite the value and refresh recency. No eviction,
            # because the size does not change.
            node.value = value
            self._unlink(node)
            self._push_front(node)
            return
        if len(self._nodes) == self._capacity:
            # Case 2: new key, cache full. The back node is the least recent entry. Remove it
            # from the list and from the map, or the map keeps a stale entry.
            victim = self._tail.prev
            if victim is None or victim is self._head:
                raise RuntimeError("a full cache always has a real node at the back")
            self._unlink(victim)
            del self._nodes[victim.key]
        # Case 3: new key, room left (or just made). Insert as the most recent entry.
        node = _Node(key, value)
        self._nodes[key] = node
        self._push_front(node)

    def __len__(self) -> int:
        return len(self._nodes)
