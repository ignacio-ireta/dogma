# Peak number of concurrent sessions

Implement `max_concurrent(sessions)`. Each session is a half-open span `(start, end)`: it is active at every whole
time `t` with `start <= t < end`. Return the largest number of sessions that are active at the same time. This is
the peak load a resource must handle, such as the number of meetings that need a room at once.

## Contract

- `sessions`: a `list` of `tuple[int, int]`, possibly empty. Each tuple is `(start, end)` with `start < end`.
  The input may be in any order and may repeat sessions.
- Two sessions where one ends at `t` and the other starts at `t` do **not** overlap. A session that ends at `t`
  is no longer active at `t`, so a session that starts at `t` does not share an active moment with it. Equivalently,
  at a shared moment, the end is processed before the start.
- Returns: an `int`, the maximum number of sessions active at one moment. `0` for an empty list.
- Raises: `ValueError` when any session has `start >= end`. The check applies to every session.
- Must not mutate `sessions`, either the list or its tuples.
- Complexity: **O(n log n) time** for `n = len(sessions)`: sort the events once, then one pass. The hidden tests
  include a list of 200,000 sessions. Counting, for each session, how many sessions contain its start does about
  `n**2` comparisons there, and it will not finish in time.

## Examples

```python
>>> max_concurrent([(1, 4), (2, 6), (3, 5)])
3
>>> max_concurrent([(1, 3), (3, 5), (5, 7)])
1
# Each session ends at the moment the next one starts, so no two are active together.
>>> max_concurrent([(0, 10), (2, 4), (3, 9), (8, 12)])
3
# At t = 3 the first three are active. At t = 8 the first, third and fourth are active. No moment has four.
```

```python
>>> max_concurrent([])
0
>>> max_concurrent([(5, 5)])
ValueError: each session needs start < end
```

## Constraints

- Allowed: built-in types (`list`, `tuple`, `int`), `len`, `sorted`, `max`, `min`, and the standard library.
- Not allowed: comparing every session with every other session. Not allowed: changing `sessions` in place.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare your answer with a brute-force count at
every event time, on small and medium random lists with shared endpoints. The edge families are: an empty list, a
single session, touching sessions in both input orders, identical sessions, nested sessions, a moment where many
sessions end and many start, a staggered chain, an invalid session (`start == end` or `start > end`) anywhere in
the list, an input that must not change, an answer that is a plain `int`, and a list of 200,000 sessions that must
finish within the time limit.
