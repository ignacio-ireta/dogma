# ROC AUC from ranks

Implement `roc_auc(y_true, scores)`. It returns the area under the ROC curve, computed as the probability that a randomly chosen positive sample scores higher than a randomly chosen negative sample, with a tie counting as one half. Compute it from ranks, the Mann-Whitney form of the statistic, in O(n log n): sort, assign ranks, sum the ranks of the positives. Do not build the ROC curve point by point.

## Contract

- `y_true`: 1-D labels, each 0 or 1 (integers or booleans). `scores`: 1-D finite real numbers. Both have the same length n.
- Returns a Python float: the fraction of (positive, negative) pairs ordered correctly, where a tied pair counts as 0.5.
- Method: rank all scores in ascending order, starting at 1. Tied scores share the average of the ranks they occupy. Let R_pos be the sum of the ranks of the positive samples, n_pos the number of positives and n_neg the number of negatives. Then AUC = (R_pos - n_pos (n_pos + 1) / 2) / (n_pos n_neg).
- Raises `ValueError` for unequal lengths, empty input, labels other than 0 and 1, non-finite scores, and inputs with no positive or no negative samples (AUC is undefined).
- Two scores are tied only when they are exactly equal as floats. No tolerance is applied, so the
  value is unchanged when every score is multiplied by the same positive constant, even 1e-9.
- Must run in O(n log n) time. Must not mutate its inputs.

## Examples

```python
>>> roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
0.75              # 3 of the 4 positive-negative pairs are ordered correctly
```

```python
>>> roc_auc([1, 0, 0, 1], [0.9, 0.9, 0.2, 0.6])
0.625             # the pair (0.9, 0.9) is tied and counts as 0.5: (0.5 + 1 + 0 + 1) / 4
```

```python
>>> roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
0.5               # every score tied: the ranking carries no information
```

On raw `breast_cancer` (569 rows), with `malignant = 1 - y` as the positive class and the single feature
`worst concave points` as the score, the value is 0.967. Its 569 scores take only 492 distinct values.

## Constraints

- Allowed: the Python standard library and NumPy.
- Not allowed: `sklearn` and `scipy`. Do not call `roc_auc_score` or `scipy.stats.rankdata`. Implement the ranking yourself.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. They compare with scikit-learn's `roc_auc_score` on tie-heavy and continuous random data, and with a direct count over all positive-negative pairs. They check the invariances: negating the scores maps AUC to 1 minus AUC, and a strictly increasing transform of the scores leaves it unchanged. They check that tied scores count as half, including all-tied data where a position-based ranking gives a wrong value, that single-class, empty, non-binary-label, NaN and mismatched-length inputs raise `ValueError`, that 300,000 scores finish quickly, and that the caller's arrays are unchanged. Realism families covered:
real `breast_cancer` features with natural ties, checked against scikit-learn, and a tiny-scale invariance
(scores multiplied by 1e-9) that rejects any absolute tolerance used to merge ties.
