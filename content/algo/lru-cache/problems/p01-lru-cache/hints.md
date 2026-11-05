## Hint 1 — concept

Two things must be fast: finding an entry by key, and knowing which entry is least recently used.
A hash map does the first. For the second, ask what an entry needs so that it can move to the
front without a search, and which end of the order holds the answer to "who goes next".

## Hint 2 — formula / approach

Keep a map from key to node, and a doubly linked list of nodes ordered from most to least recent.
Use a sentinel node at each end. A get hit unlinks the node and pushes it to the front. A put of an
existing key does the same after it overwrites the value. A put of a new key checks the size first:
when the cache is full, remove the node before the back sentinel from the list and from the map,
then push the new node to the front.

## Hint 3 — code shape

Write `_unlink(node)` and `_push_front(node)` first and test them alone. They touch four pointers
each. Then `get` is a lookup, an unlink and a push. `put` has three branches: an existing key, a
full cache, and a free slot. Check the size before you insert, not after.
