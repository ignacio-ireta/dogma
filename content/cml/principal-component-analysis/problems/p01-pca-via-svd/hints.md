## Hint 1: concept

The principal axes are the directions in which the centred data spreads most. Each axis has a variance,
and the axes are ranked by that variance. The singular value decomposition of the centred matrix gives
the axes directly, already sorted. Centre the columns before anything else, because spread is measured
around the mean.

## Hint 2: formula / approach

Write the centred data as $\tilde X = U S V^\top$. The rows of $V^\top$ are the axes. The explained variance
of axis $i$ is $s_i^2/(n-1)$, where $s_i$ is its singular value. The ratio divides by the total variance of
all $d$ columns, not by the variance of the kept axes. Each kept axis then needs a sign chosen by the rule
in the contract. The scores are the centred data projected onto the signed axes.

## Hint 3: code shape

Validate first: 2-D, at least two rows, finite values, and an integer `n_components` that is not a
boolean and lies in range. Then:

```python
centred = x - x.mean(axis=0)
_, s, vt = np.linalg.svd(centred, full_matrices=False)
# for each kept row: find the first index whose magnitude is within a relative 1e-9 of the row's
# largest magnitude, and multiply the row by the sign of that entry
# guard the zero-total-variance case before you divide
```
