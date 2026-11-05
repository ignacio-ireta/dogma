# Lazy k-way merge of sorted streams

Implement `merge_sorted_streams(streams)`. `streams` is a sequence of `k` iterables. Each stream yields
`(key, label)` pairs whose keys are non-decreasing. Return an iterator of all the pairs, in non-decreasing
order of key. The streams can be generators, they can be infinite, and only the part the caller consumes
may be read. The merge uses a heap, so that a single pass over `n` pairs costs O(n log k).

## Contract

- `streams`: a sequence of `k >= 0` iterables. Each yields `(key, label)` pairs with non-decreasing keys.
  Keys are mutually comparable (ints or floats). Labels can be any objects, and they are **never compared**
  by the merge.
- Returns an iterator (a generator is fine). **Calling the function reads nothing from any stream.** The
  first `next()` on the result is the first point at which a stream may be read.
- Ordering: the output is in non-decreasing key order. Among equal keys, a pair from a stream with a lower
  index in `streams` comes first. Pairs from the same stream keep their own order.
- Laziness: after the caller has taken `m` pairs in total, a stream that has yielded `t` of its pairs has
  been read at most `t + 1` times, so at most one pair per stream is read ahead. Infinite streams are
  fine as long as the caller stops.
- An empty `streams` gives an empty iterator. An empty stream contributes nothing.
- Runs in O(n log k) time for `n` pairs in total, using a heap of the current head of each stream.
- Must not mutate its inputs.

## Examples

```python
>>> list(merge_sorted_streams([[(1, "a"), (4, "b")], [(2, "c"), (4, "d")]]))
[(1, 'a'), (2, 'c'), (4, 'b'), (4, 'd')]
# equal keys 4: 'b' is from stream 0 and 'd' from stream 1, so 'b' comes first
```

```python
>>> evens = ((k, "even") for k in itertools.count(0, 2))
>>> odds = ((k, "odd") for k in itertools.count(1, 2))
>>> [key for key, _ in itertools.islice(merge_sorted_streams([evens, odds]), 6)]
[0, 1, 2, 3, 4, 5]
# two infinite streams: only the first six pairs were read
```

```python
>>> list(merge_sorted_streams([[], iter([(3, "x")]), ()]))
[(3, 'x')]
```

## Constraints

- Allowed: the Python standard library, in particular `heapq.heappush`, `heapq.heappop` and
  `heapq.heapify`.
- Not allowed: `heapq.merge` (listed in `banned_imports`), and any approach that reads every stream to
  the end before yielding the first pair.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare with a stable sort of the
concatenated streams on random finite inputs with heavy ties and empty streams. They check that equal keys
come out in stream order, even when the labels would sort the other way, and that labels which cannot be
compared (dictionaries) work with equal keys. They check that calling the function reads nothing, that
infinite streams merge correctly, and that each stream is read at most one pair beyond what the caller
took, through a read counter on every stream. They check empty and single streams, generator inputs, and a
large merge of 5,000 streams that must finish within a time limit. A linear scan over the stream heads
passes the answers but fails that time limit, and an eager merge fails the read budget.
