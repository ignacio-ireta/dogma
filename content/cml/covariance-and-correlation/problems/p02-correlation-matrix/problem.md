# Correlation matrix

Implement `correlation_matrix(X)`, which returns the Pearson correlation matrix of a data matrix. Each
row of `X` is one observation and each column is one variable. Correlation is covariance rescaled to
$[-1, 1]$: entry $[j,k]$ is $\dfrac{C_{jk}}{\sqrt{C_{jj}\,C_{kk}}}$, where $C$ is the sample covariance
with divisor $n-1$. Because the scale cancels, correlation does not depend on the units of each column.

## Contract

- `X`: 2-D array-like of real numbers, shape `(n, d)` with $n \ge 2$ rows and $d \ge 1$ columns.
  Integers allowed. Every entry is finite.
- Returns: NumPy array of shape `(d, d)` and dtype `float64`. Every entry is in $[-1, 1]$ (clip any
  rounding overshoot). Non-constant columns have a diagonal entry of 1.
- **Constant columns.** A column is constant when all its entries are equal. Its correlations are
  undefined, so every entry in its row and its column is `NaN`, including its diagonal entry. No
  exception is raised for a constant column. Other entries are still computed normally.
- Raises `ValueError` when `X` is not 2-D, when it has no rows or no columns, when it has fewer than
  two rows, or when it contains NaN or infinity.
- Must not mutate its input.

## Examples

```python
>>> correlation_matrix([[1, 3], [2, 1], [3, 2]])
array([[ 1. , -0.5],
       [-0.5,  1. ]])
>>> correlation_matrix([[1, 2], [2, 4], [3, 6]])         # column 2 = 2 * column 1
array([[1., 1.],
       [1., 1.]])
>>> correlation_matrix([[1, 5], [2, 5], [3, 5]])         # column 2 is constant
array([[ 1., nan],
       [nan, nan]])
>>> correlation_matrix([[1.0, 2.0]])
ValueError: need at least two observations to measure correlation
```

## Constraints

- Allowed: NumPy array operations (`np.asarray`, `np.where`, `np.clip`, `np.sqrt`, `@`, `np.outer`,
  `np.argsort`, ...).
- Not allowed: `numpy.cov`, `numpy.corrcoef`, `pandas` and `scipy`. Build the covariance and the
  scaling yourself.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with
`numpy.corrcoef` on random data; a unit diagonal, symmetry and the $[-1, 1]$ bound; exact correlation
for perfectly linear columns; invariance to rescaling any column by a positive factor, including factors
near $10^{200}$ and $10^{-200}$ where a naive norm overflows or underflows; the constant-column convention,
including a column whose repeated value is not exactly representable (the mean of the repeated values
differs from the value itself); a column with a tiny spread or a large offset that is not constant; fewer
than two rows, non-2-D, NaN, infinite, empty and zero-variable input raising `ValueError`; integer input;
and inputs left unchanged.
