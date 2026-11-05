# Precision, recall, reciprocal rank and AP at k

Implement `retrieval_metrics(ranked_ids, relevant_ids, k)`. A retriever returned `ranked_ids`, best
first, for one query. `relevant_ids` are the ids that a human judged relevant to that query. Return the
four numbers a retrieval evaluation reports for one query: precision at k, recall at k, reciprocal rank
and average precision at k.

## Contract

- `ranked_ids`: a sequence of hashable ids, best first. It may be empty. No id may appear twice.
- `relevant_ids`: an iterable of hashable ids. Duplicates are ignored. It must contain at least one id.
- `k`: an integer `k >= 1`. `k` may be larger than the length of `ranked_ids`.
- Let `top` be the first `k` entries of `ranked_ids` (fewer if the list is shorter), and let `R` be the
  set of relevant ids.
- `precision_at_k` = (number of relevant ids in `top`) / `k`. The denominator is `k` even when the list
  has fewer than `k` entries.
- `recall_at_k` = (number of relevant ids in `top`) / |R|.
- `reciprocal_rank` = 1 / (1-based position of the first relevant id in the **whole** ranked list), or
  `0.0` if no relevant id is ranked. It does not depend on `k`.
- `average_precision_at_k`: for each 1-based position `i <= min(k, n)` whose item is relevant, add
  precision at `i`, which is (relevant items among the first `i`) / `i`. Divide the sum by
  `min(k, |R|)`. Relevant ids that are never ranked add nothing to the sum, and the divisor still counts
  them.
- Returns: a dict with exactly the keys `precision_at_k`, `recall_at_k`, `reciprocal_rank` and
  `average_precision_at_k`, each a Python float.
- Raises `ValueError` for: a repeated id in `ranked_ids`, an empty `relevant_ids`, and a `k` that is not
  an integer of at least 1.
- Must not mutate its inputs.

## Examples

Values are rounded to four places in the examples below; your function returns full floats.

```python
>>> retrieval_metrics(["d3", "d1", "d7", "d2"], ["d1", "d2", "d9"], k=3)
{'precision_at_k': 0.3333, 'recall_at_k': 0.3333, 'reciprocal_rank': 0.5, 'average_precision_at_k': 0.1667}
# top 3 = d3, d1, d7: one hit at position 2. AP sums precision 1/2 there, divided by min(3, 3).
```

```python
>>> retrieval_metrics(["a", "b", "c"], ["a", "c"], k=5)
{'precision_at_k': 0.4, 'recall_at_k': 1.0, 'reciprocal_rank': 1.0, 'average_precision_at_k': 0.8333}
# the list has 3 entries but k is 5: precision still divides by 5. AP = (1 + 2/3) / min(5, 2).
```

```python
>>> retrieval_metrics(["x", "y"], ["z"], k=2)
{'precision_at_k': 0.0, 'recall_at_k': 0.0, 'reciprocal_rank': 0.0, 'average_precision_at_k': 0.0}
# nothing relevant was retrieved: every metric is 0.0, and no error is raised
```

## Constraints

- Allowed: the Python standard library.
- Not allowed: `sklearn` (listed in `banned_imports`). Count the hits yourself.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare the four values with an
independent NumPy computation on random rankings, with `k` both smaller and larger than the list. They
check the denominators: precision divides by `k` even for a short list, and average precision divides by
`min(k, |R|)`, so it reaches 1.0 when the top `k` are all relevant even though `|R|` is larger than `k`.
They check that reciprocal rank is not cut at `k`, that duplicate relevant ids are counted once, and that
appending irrelevant items to the ranking changes no metric. They check that a repeated ranked id, an
empty relevant set and a `k` of 0 raise `ValueError`, that an empty ranking gives all zeros, that a
perfect ranking gives all ones, and that the inputs are unchanged. Ids are large and non-sequential
integers and strings.
