# NDCG at k for one query

Implement `ndcg_at_k(true_relevance, scores, k=None)`. It scores one ranked list against graded
relevance judgements. The items are ordered by `scores`, each item's relevance is its gain, and a
gain counts less the deeper it sits in the list. The result is the normalised discounted cumulative
gain (NDCG): a number in [0, 1], where 1 means the list is as good as the best possible order of the
same items.

## Contract

- `true_relevance`: 1-D, length n >= 1, finite non-negative real numbers (0 means irrelevant).
  `scores`: 1-D, length n, finite real numbers. A higher score means the item is ranked earlier.
- `k`: `None` for the whole list, or an integer `k >= 1`. A `k` larger than n means the same as
  `k = n`.
- Ranking: sort the items by score, descending. The item at 1-based position `r` has gain equal to its
  relevance and discount `1 / log2(r + 1)`.
- **Ties.** Items with equal scores have no defined order. The value is the expectation over a
  uniformly random order inside each tie group, which is what `sklearn.metrics.ndcg_score` returns by
  default. A tie group that occupies positions a..b contributes the mean relevance of its members times
  the sum of the discounts of those positions that are within k. Do not break ties by input position.
- DCG@k is the sum of gain times discount over the first `min(k, n)` positions. IDCG@k is the same sum
  for the items sorted by relevance, descending. NDCG@k = DCG@k / IDCG@k.
- If IDCG@k is 0 (every relevance is 0), return `0.0`.
- Returns: a Python float.
- Raises `ValueError` for: an input that is not one-dimensional, unequal lengths, empty input, a
  negative or non-finite relevance, a NaN or infinite score, and a `k` that is neither `None` nor an
  integer of at least 1.
- Must run in O(n log n) time. Must not mutate its inputs.

## Examples

```python
>>> ndcg_at_k([0, 1], [0.2, 0.9])
1.0               # the relevant item is ranked first, so the list is ideal
```

```python
>>> ndcg_at_k([0, 3, 1], [0.7, 0.7, 0.2])
0.8115            # the first two items tie: their mean gain 1.5 takes positions 1 and 2
```

```python
>>> ndcg_at_k([1, 3, 2], [0.9, 0.8, 0.1], k=2)
0.6788            # only the top two positions count: DCG = 1 + 3/log2(3), IDCG = 3 + 2/log2(3)
```

```python
>>> ndcg_at_k([0, 0], [0.3, 0.1])
0.0               # no relevant item at all: IDCG is 0, so the value is defined as 0.0
```

## Constraints

- Allowed: the Python standard library and NumPy.
- Not allowed: `sklearn` (listed in `banned_imports`). Do not call `ndcg_score`. Implement the ranking
  and the tie rule yourself.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They compare with scikit-learn's
`ndcg_score` on tie-heavy and continuous scores, with and without `k`. They compare with a brute-force
average over every order of the tied items on small inputs. They check that the value does not depend
on the order of the input items, that scaling the relevance changes nothing, that a strictly increasing
transform of the scores changes nothing, and that an all-tied list gives the averaged value. They check
that all-zero relevance gives 0.0, that a single item is valid, that `k` larger than n equals the full
list, and that negative relevance, NaN or infinite scores, `k = 0`, empty input, mismatched lengths and
a matrix input raise `ValueError`. A 200,000-item list must finish quickly, and the caller's arrays must
be unchanged.
