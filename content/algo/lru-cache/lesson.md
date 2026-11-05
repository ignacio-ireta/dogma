---
title: LRU cache
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01 by writing the four cases of put in comments first."
---

## The system

An LRU (least recently used) cache holds at most $C$ entries. When it is full and a new key
arrives, it evicts the entry that was used longest ago. "Used" means a successful `get` or a
`put` of that key. Two structures do the work, and each has one job:

| Structure | Job | Operation cost |
|---|---|---|
| Hash map: key → node | Find an entry by key | O(1) expected |
| Doubly linked list of nodes | Keep entries in recency order: front = most recent, back = least recent | Unlink or insert a known node: O(1) |

The map finds a node, and the list reorders it. Prev and next pointers let a node unlink
itself without a search.

**Sentinel nodes.** The list has a `head` and a `tail` node that hold no data. The real
entries sit between them. Every insert and unlink then has a neighbour on both sides, so there
are no special cases for an empty list or for the first and last entry.

**The rules.** Let $n$ be the number of stored entries, with $0 \le n \le C$:

| Operation | Hit or case | Effect |
|---|---|---|
| `get(k)` | $k$ stored | Move its node to the front; return its value |
| `get(k)` | $k$ not stored | Return `None`; change nothing |
| `put(k, v)` | $k$ stored | Overwrite the value, move the node to the front; no eviction |
| `put(k, v)` | $k$ new, $n < C$ | Insert a node at the front, add it to the map |
| `put(k, v)` | $k$ new, $n = C$ | Remove the back node (least recent) from the list and the map, then insert as above |

The invariant: the map and the list hold the same $n$ entries, and reading the list from the
front gives recency order.

## Worked example

Capacity $C = 2$. The list is shown as front → back.

| Step | Operation | Result | List (front → back) |
|---|---|---|---|
| 1 | put(1, a) | new, $n<2$ | 1 |
| 2 | put(2, b) | new, $n<2$ | 2, 1 |
| 3 | get(1) | hit, returns a | 1, 2 |
| 4 | put(3, c) | new, $n=2$: evict back node 2 | 3, 1 |
| 5 | get(2) | miss, returns None | 3, 1 |
| 6 | put(1, A) | existing: overwrite, move to front, no eviction | 1, 3 |
| 7 | put(4, d) | new, $n=2$: evict back node 3 | 4, 1 |

Step 3 is the one people skip. Without the refresh on `get`, step 4 evicts key 1, which was
used more recently than key 2.

## Why it works

Each operation is one hash lookup and a fixed number of pointer changes. The victim is always
`tail.prev`, so eviction is O(1) too.

**When OrderedDict is acceptable.** `collections.OrderedDict` is this structure: `move_to_end(k)`
refreshes recency and `popitem(last=False)` evicts the least recent entry, both in O(1). In
production, use it and say so. When the prompt bans it, build the list: the interviewer wants
the two-structure invariant, not a call that hides it.

`functools.lru_cache` memoises function calls. It has no explicit `put`, so it is no substitute.

## Signal vs. noise

- **Signal:** the recency rule has four cases in `put` and one in `get`. Name each case, and
  name the one that does not evict.
- **Signal:** the sentinel nodes remove boundary checks. Mention them before writing pointer code.
- **Noise for now:** thread safety, memory layout, and whether nodes use `__slots__`. Mention
  thread safety as a follow-up.

## Where it breaks

- **No refresh on get.** The cache becomes FIFO: it evicts the oldest insertion, not the least
  recently used key.
- **Overwrite without a move.** A `put` of an existing key updates the value but leaves the
  node in place, so a hot key is evicted early.
- **Eviction without removing from the map.** A later `get` returns a stale value.
- **Removing from the list by search.** `list.remove` costs O(C) per operation.
- **Evicting before checking the key.** Eviction on an overwrite removes a live entry for no
  reason. Check existence first.
- **The dict re-insert trick.** `pop` and re-insert moves a key to the end of a CPython dict,
  but finding the oldest key can scan deleted slots first. Choose the list when the bound must
  hold.

## Interview angle

Say the two structures first: a map for lookup, a doubly linked list for recency, with sentinel
nodes at both ends. Then state the four cases of `put`, and note that `get` on a hit also
refreshes recency. Finish with the cost: O(1) expected for both operations. If the interviewer
asks about thread safety, name a single lock around both methods as the first step, and say what
it costs.

## Connections

The prerequisite is the hash map: the lookup is a dictionary from key to node. The same
hash-plus-list pair gives O(1) removal from any ordered collection. Caches of this shape appear
in page caches and model serving.
