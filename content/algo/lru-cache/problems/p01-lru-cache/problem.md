# LRU cache

Implement a class `LRUCache` that stores at most `capacity` key-value pairs. When a new key
arrives at a full cache, it evicts the least recently used entry. An entry is used when a `get`
returns its value, or when a `put` stores a value under its key.

Build the recency order yourself. You need a hash map from key to node, and a doubly linked list
whose front holds the most recent entry and whose back holds the least recent. Use sentinel head
and tail nodes so that insert and removal have no empty-list cases.

## Contract

- `LRUCache(capacity)`: `capacity` must be a Python `int` of at least 1. Raises `ValueError` for
  anything else, including `bool`, `float` and `None`.
- `get(key) -> value | None`: if `key` is stored, mark it as the most recently used entry and
  return its value. If it is not stored, return `None` and change nothing.
- `put(key, value) -> None`: `value` must not be `None`, because `None` is the miss result of
  `get`. Raises `ValueError` for `None`. If `key` is stored, overwrite its value and mark it as
  most recently used, with no eviction. If `key` is new and the cache is full, evict the least
  recently used entry first, then store the new entry as the most recently used.
- `len(cache)`: the number of stored entries. It is never greater than `capacity`.
- Keys are any hashable Python values. Values are any Python object except `None`.
- `get` and `put` must run in O(1) expected time, with no scan over the stored entries.
- A removed entry is gone: a later `get` of its key returns `None`, and a later `put` of that key
  stores a new entry.

## Examples

```python
>>> cache = LRUCache(2)
>>> cache.put(1, "a"); cache.put(2, "b")
>>> cache.get(1)
'a'                       # 1 is now the most recent entry; the order is 1, 2
>>> cache.put(3, "c")       # full: evict the least recent entry, which is 2
>>> cache.get(2) is None
True
>>> cache.get(3)
'c'
>>> cache.put(1, "A")       # existing key: overwrite, no eviction
>>> len(cache)
2
>>> cache.get(1)
'A'
>>> LRUCache(0)
ValueError                # capacity must be at least 1
>>> LRUCache(1).put("k", None)
ValueError                # None cannot be stored
```

## Constraints

- Allowed: built-in types, plain classes, and `typing` for annotations.
- Not allowed: the whole `collections` package, including `collections.OrderedDict`, and the
  whole `functools` package, including `functools.lru_cache`. A plain `dict` is fine. The point
  is to build the recency list yourself.
- A cache that stores entries in a list and removes the victim by searching the list is
  correct on small inputs and fails the time test on large ones.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare every answer and every
length against an independent model over seeded random operation sequences, with capacities from
one upward and key spaces that force many evictions. The edge families they cover: capacity one; a
get miss that must not change the order; a get hit that must protect its entry from eviction; a
put of an existing key that must refresh recency and must not evict; a removed entry that must not
come back with its old value; invalid capacities; the `None` value rule; the size bound; and a
100,000-operation sequence (100,000 fills, then 100,000 mixed operations) that must finish
within 5 seconds. A correct linked-list cache does this in well under a second.

The hidden tests draw their bulk inputs from the real dataset `online_retail` (see
`docs/DATASETS.md`).
