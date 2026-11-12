# Elastic net by coordinate descent

Implement `elastic_net_cd(X, y, alpha, l1_ratio, *, max_iter=10_000, tol=1e-8)`. It minimises the elastic
net objective

    (1/(2n)) ‖y − Xw‖² + α ρ ‖w‖₁ + ½ α (1 − ρ) ‖w‖²,    ρ = `l1_ratio`,

over w, by cyclic coordinate descent. As in the lasso, there is **no intercept**: fit y ≈ Xw exactly as
given. With ρ = 1 the problem is the lasso. With ρ = 0 it is ridge regression.

## Contract

- `X`: 2-D array-like of shape (n, d), n ≥ 1, d ≥ 1, finite real numbers. Integers allowed.
- `y`: 1-D array-like of shape (n,), finite real numbers.
- `alpha`: a finite real number ≥ 0.
- `l1_ratio`: a finite real number in [0, 1].
- `max_iter`: integer ≥ 1. `tol`: a finite real number > 0.
- Returns: float64 array of shape (d,).
- Algorithm. Start from the all-zero coefficient vector. With ρⱼ = (1/n) Xⱼᵀ (y − Σ_{k≠j} Xₖ wₖ) and zⱼ = (1/n) ‖Xⱼ‖², update
  wⱼ ← S(ρⱼ, α ρ) / (zⱼ + α (1 − ρ)), where S(x, λ) = sign(x) · max(|x| − λ, 0). If the denominator is
  zero, keep wⱼ = 0. One sweep updates j = 0, 1, …, d − 1 in order, using the newest values. Stop after a
  sweep whose largest change |Δwⱼ| is below `tol`, or after `max_iter` sweeps.
- Raises `ValueError` when X is not 2-D; when y is not 1-D or its length differs from the rows of X; when n
  or d is 0; when X or y is non-finite; when alpha is negative or non-finite; when `l1_ratio` is outside
  [0, 1] or non-finite; when max_iter < 1; or when tol is not a finite number greater than 0.
- Must not mutate inputs.

## Examples

```python
>>> elastic_net_cd([[1.], [2.], [3.]], [2., 4., 6.], 1.0, 0.5)
array([1.70967742])   # (28/3 - 0.5) / (14/3 + 0.5) = 53/31
>>> elastic_net_cd([[1.], [2.], [3.]], [2., 4., 6.], 1.0, 1.0)
array([1.78571429])   # l1_ratio = 1 is the lasso: (28/3 - 1) / (14/3) = 25/14
>>> elastic_net_cd([[1.], [2.], [3.]], [2., 4., 6.], 1.0, 1.5)
ValueError            # l1_ratio must lie in [0, 1]
```

## Constraints

- Allowed: NumPy array arithmetic (sums, products, `np.sign`, `np.maximum`, `np.abs`).
- Not allowed: `sklearn`, `scipy`, and `numpy.linalg`. Coordinate descent needs no matrix factorisation.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare the coefficients with
scikit-learn's `ElasticNet` (no intercept) over several mixing ratios, and check the KKT conditions at the
returned point. They check the ρ = 0 limit against the ridge closed form, and the threshold for all zeros.
They check the grouping effect: duplicated and nearly duplicated columns share their weight evenly, where
the lasso picks one of them. They also check a zero column with ρ = 1 (which has zero denominator), the
scaling of y with α, invalid inputs, integer input, and unchanged inputs.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `diabetes` (see
`docs/DATASETS.md`).
