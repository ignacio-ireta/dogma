# Reservoir sample of k items

Implement `reservoir_sample(stream, k, rng)`. It returns a uniform random sample of `k` items from
`stream`, which may be any iterable (a list, a `range` or a generator). The stream is read in one
pass and its length is not known in advance. The sample is built with Algorithm R.

## Contract

- `stream`: any iterable of items. Read it once, in order, to its end. Do not call `len` on it, and
  do not store the whole stream: memory must stay O(k).
- `k`: an int. `k = 0` returns `[]`, whatever the stream and the draws. `k < 0` is an error.
- `rng`: a NumPy `Generator`. The draws follow this protocol exactly:
  1. The item with 0-based index `i < k` goes to slot `i`, in arrival order.
  2. For each item with index `i >= k`, make one draw `j = rng.integers(0, i + 1)`, which is
     uniform over `0, ..., i` inclusive. If `j < k`, the item replaces slot `j`. Otherwise the item
     is discarded.
  3. No other random draw is made.
- Returns: a `list` of length `min(k, number of items)`, in slot order: element `j` is the item in
  slot `j`. A stream shorter than `k` gives every item, in arrival order.
- Raises `ValueError` when `k < 0`.
- Must not mutate the stream's items.

## Examples

```python
>>> reservoir_sample(range(10), 3, np.random.default_rng(0))
[9, 6, 7]
>>> reservoir_sample(range(3), 5, np.random.default_rng(0))
[0, 1, 2]                          # a stream shorter than k: every item, in order
>>> reservoir_sample(range(5), -1, np.random.default_rng(0))
ValueError                         # k must be non-negative
```

## Constraints

- Allowed: the standard library (`itertools`, `math`, `typing`) and the `rng` you are given.
- Not allowed: `scipy`. Use `rng.integers` for every random choice. Other generators give other
  draws, and the protocol fixes the draws.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: exact agreement with an
independent two-pass version of the same protocol, for random stream lengths, values of `k` and
seeds; one `rng.integers(0, i + 1)` draw per item after the first `k`, and no other draws; uniform
inclusion of each item over many seeds (a chi-square test); uniform choice of every `k`-subset; a
generator input read to its end and giving the same sample as a list; a stream shorter than `k`;
an empty stream; `k = 0` and a negative `k`; memory that stays bounded by `k` on a long generator,
measured with `tracemalloc`; the length and membership of the sample; and no mutation of the
input.

The hidden tests draw their bulk inputs from the real dataset `gutenberg_frankenstein` (see
`docs/DATASETS.md`).
