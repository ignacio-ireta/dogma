# PCA via the SVD

Implement `pca(X, n_components)`, which runs principal component analysis on a data matrix. Each row
of `X` is one observation and each column is one variable. Centre the columns, take the singular
value decomposition of the centred data with `numpy.linalg.svd`, and return the top `n_components`
principal axes with their explained variances and explained-variance ratios, together with the
observations projected onto those axes.

## Contract

- `X`: 2-D array-like of real numbers, shape `(n, d)` with `n >= 2` and `d >= 1`. Integers allowed.
  Every entry is finite.
- `n_components`: an `int` with `1 <= n_components <= min(n, d)`. Booleans are rejected.
- Centre each column by its mean first. The means are not returned.
- Returns: a `dict` with exactly these four keys. Each value is a float64 array, and `k` is
  `n_components`.
  - `"components"`: shape `(k, d)`. Row `i` is the unit-length `i`-th principal axis. Rows are ordered
    by decreasing explained variance.
  - `"explained_variance"`: shape `(k,)`. The sample variance (divisor `n - 1`) of the data projected
    onto each axis.
  - `"explained_variance_ratio"`: shape `(k,)`. Each explained variance divided by the total variance,
    which is the sum over all `d` columns of their sample variances (divisor `n - 1`). Dividing by the
    variance of the kept axes only is wrong. If the total is zero, because every column is constant,
    every ratio is `0.0`. Zero means exactly zero: a tiny but positive total is not constant, and its ratios
  are computed as usual.
  - `"transformed"`: shape `(n, k)`, equal to the centred `X` times `components.T`.
- Sign convention: in each row of `components`, the entry of largest absolute value is positive. This
  is the rule scikit-learn applies in `PCA(svd_solver="full")`, its `svd_flip` with
  `u_based_decision=False`. If two or more entries lie within a relative `1e-9` of the largest absolute
  value, the lowest index decides. scikit-learn settles such exact ties by floating-point rounding, so
  it can differ from this rule on them.
- Raises `ValueError` when `X` is not 2-D, has fewer than two rows or no columns, contains NaN or
  infinity, or when `n_components` is not an integer in range.
- Must not mutate its input.

## Examples

```python
>>> r = pca([[-3, -1], [0, -2], [0, 3]], n_components=1)
>>> np.round(r["components"], 4)
array([[0.3162, 0.9487]])
>>> r["explained_variance"], r["explained_variance_ratio"]    # up to rounding
(array([7.5]), array([0.75]))
>>> np.round(r["transformed"].ravel(), 4)
array([-1.5811, -1.5811,  3.1623])
>>> pca([[1, -1], [-1, 1], [0, 0]], n_components=1)["components"]   # exact tie: lowest index is positive
array([[ 0.70710678, -0.70710678]])
>>> pca([[1, 2], [3, 4]], n_components=3)     # more components than min(n, d) = 2
ValueError
```

On raw `wine` (178 rows, 13 columns; column variances from about 0.02 to about 10^5, with proline the
largest), the first axis carries about 99.8% of the variance. `explained_variance_ratio[0]` is about 0.998 with
`n_components=1`, and the axis is almost pure proline. The same data scaled by $2^{-40}$ has a total variance near
$10^{-19}$, and its ratios must be unchanged.

## Constraints

- Allowed: NumPy array operations, including `numpy.linalg.svd` and `numpy.linalg.eigh`.
- Not allowed: `sklearn` and `scipy`. Compute the decomposition yourself.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with
scikit-learn's full-SVD PCA on random data, including signs; that the explained variances are the
sample variances of the projections; that the ratios use the total variance; shift invariance, which
catches a missing centring step; orthonormal axes in decreasing order of variance; the sign rule on its
own; that keeping every axis preserves distances from the mean; rank-one and constant data; integer
input; invalid input and parameters; and inputs left unchanged. The realism families are mixed-scale real data (raw
`wine`, where one column holds nearly all the variance), digit-style data with constant columns and a rank below
its width, and the same real data scaled by a power of two, where a tiny total variance must still give its real
ratios.

The hidden tests draw their bulk inputs from the real datasets `diabetes` and `wine` (see
`docs/DATASETS.md`).
