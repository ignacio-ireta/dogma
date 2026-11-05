from __future__ import annotations

from typing import Any


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
        raise NotImplementedError

    def get(self, key: Any) -> Any | None:
        """Return the value stored under key and mark it most recent, or None on a miss."""
        raise NotImplementedError

    def put(self, key: Any, value: Any) -> None:
        """Store value under key as the most recent entry, evicting the least recent if full.

        Raise ValueError when value is None.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of stored entries."""
        raise NotImplementedError
