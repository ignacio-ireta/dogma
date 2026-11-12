# Lasso by coordinate descent

Implement `lasso_cd(X, y, alpha, *, max_iter=10_000, tol=1e-8)`. It minimises the lasso objective

    (1/(2n)) ‖y − Xw‖² + α ‖w‖₁

over w, by cyclic coordinate descent with soft-thresholding. **There is no intercept.** The function fits
y ≈ Xw exactly as given, so a caller who wants an intercept must centre X and y first. Do not add one.

## Contract

- `X`: 2-D array-like of shape (n, d), n ≥ 1, d ≥ 1, finite real numbers. Integers allowed.
- `y`: 1-D array-like of shape (n,), finite real numbers.
- `alpha`: a finite real number ≥ 0.
- `max_iter`: integer ≥ 1, the most full sweeps over the coordinates.
- `tol`: a finite real number > 0.
- Returns: float64 array of shape (d,).
- Algorithm. Start from the all-zero coefficient vector. For each coordinate j, let ρⱼ = (1/n) Xⱼᵀ (y − Σ_{k≠j} Xₖ wₖ) and zⱼ = (1/n) ‖Xⱼ‖². Then
  wⱼ ← S(ρⱼ, α) / zⱼ, where S(x, λ) = sign(x) · max(|x| − λ, 0) is the soft threshold.
  A column with zⱼ = 0 keeps wⱼ = 0. One sweep updates j = 0, 1, …, d − 1 in that order, using the newest
  values. Stop after a sweep whose largest change |Δwⱼ| is below `tol`, or after `max_iter` sweeps.
- Raises `ValueError` when X is not 2-D; when y is not 1-D or its length differs from the rows of X; when
  n or d is 0; when X or y is non-finite; when alpha is negative or non-finite; when max_iter < 1; or when
  tol is not a finite number greater than 0.
- Must not mutate inputs.

## Examples

```python
>>> lasso_cd([[1.], [2.], [3.]], [2., 4., 6.], 1.0)
array([1.78571429])   # (rho - alpha) / z = (28/3 - 1) / (14/3) = 25/14
>>> lasso_cd([[1.], [2.], [3.]], [2., 4., 6.], 10.0)
array([0.])           # alpha above rho = 28/3 sets the coefficient to exactly zero
>>> lasso_cd([[1.], [2.], [3.]], [2., 4., 6.], -1.0)
ValueError            # alpha must be >= 0
```

## Constraints

- Allowed: NumPy array arithmetic (sums, products, `np.sign`, `np.maximum`, `np.abs`).
- Not allowed: `sklearn`, `scipy`, and `numpy.linalg`. The method needs no matrix factorisation or solve.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare the coefficients with
scikit-learn's `Lasso` (no intercept) on random data, and check the KKT optimality conditions at the
returned point. They also check the sparsity thresholds: all zeros at and above α_max = max |Xⱼᵀy| / n, and
exactly one nonzero just below it. They check that the ℓ₁ norm does not grow with α and that small α recovers
least squares. They cover uncentred data (no hidden intercept), zero columns, exactly zero correlations (which
must give exact zeros, not NaN), wide designs, scaling
of y with α, invalid inputs, integer input, and unchanged inputs.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `diabetes` (see
`docs/DATASETS.md`).
