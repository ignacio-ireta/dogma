# Interview notes — LRU cache

## Q1. Design a cache with a fixed capacity that evicts the least recently used entry.
**60-second answer.** Two structures. A hash map from key to node gives O(1) lookup. A doubly
linked list orders the nodes by recency: the front is the most recent, and the back is the least
recent. Sentinel nodes at both ends remove boundary cases. A get hit moves its node to the front.
A put either overwrites an existing node and moves it to the front, or inserts a new node at the
front and, when the cache is full, first removes the back node from both structures. Both
operations are O(1) expected.
**Follow-ups.** Why a doubly linked list? A node must unlink itself in O(1), which needs a prev
pointer. Why not OrderedDict? It is this structure, and in production it is a good choice; in an
interview, build the list to show the invariant. Is it thread-safe? No. Add one lock around both
methods first, then discuss contention.

## Q2. What is the cost of a get that misses?
**60-second answer.** O(1) expected: one map lookup, and no list change. Returning None on a miss
is a contract choice: it is only unambiguous when stored values cannot be None.
**Follow-ups.** What if values can be None? Use a separate sentinel object or a found flag.
What if the keys are unhashable? The map rejects them, so the contract must require hashable keys.

## Common mistakes
- Forgetting the refresh on get, which turns the cache into FIFO.
- Overwriting a value without moving its node to the front.
- Removing the victim from the list but not from the map.
- Scanning the list to find the victim, which makes each operation O(C).
