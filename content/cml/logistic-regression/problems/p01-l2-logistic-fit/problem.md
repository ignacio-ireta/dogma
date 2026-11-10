# L2-regularised logistic regression

Fit a binary logistic regression by minimising the penalised mean log-loss. With $n$ samples in rows of
$X \in \mathbb{R}^{n \times d}$, labels $y \in \{0, 1\}^n$, intercept $b$ and coefficients $\mathbf{w} \in \mathbb{R}^d$,
the linear score is $\eta_i = b + \mathbf{x}_i^\top \mathbf{w}$ and the probability is $p_i = \sigma(\eta_i)$.
Implement `fit_logistic_regression(X, y, *, l2=1.0, max_iter=100, tol=1e-10)`, which returns the pair
$(b, \mathbf{w})$ that minimises

$$J(b, \mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n}\Big[\log\big(1 + e^{\eta_i}\big) - y_i\,\eta_i\Big] + \frac{\ell_2}{2}\,\lVert \mathbf{w}\rVert_2^2.$$

The intercept $b$ is **not** penalised. The bracketed term is the Bernoulli log-loss, written in a form that
does not need $\log \sigma$. Its gradient is $\partial J/\partial b = \frac1n \sum_i (p_i - y_i)$ and
$\nabla_{\mathbf{w}} J = \frac1n X^\top(\mathbf{p} - \mathbf{y}) + \ell_2 \mathbf{w}$.

## Contract

- `X`: 2-D array-like, shape `(n, d)`, with $n \ge 1$ and $d \ge 1$. Every entry is finite.
- `y`: 1-D array-like, shape `(n,)`. Every entry is exactly `0` or `1`. Both classes must appear.
- `l2`: finite real number, $\ell_2 \ge 0$. Zero is allowed; see the note below.
- `max_iter`: integer, at least 1. `tol`: finite number greater than 0.
- Returns: `(intercept, coef)`. `intercept` is a Python `float`. `coef` is a `float64` array of shape `(d,)`.
  At return, the largest absolute entry of $\nabla J$ is at most `tol`.
- Raises: `ValueError` for any violation of the input rules above. `RuntimeError` when the gradient
  condition is not reached within `max_iter` iterations.
- Note: with `l2 = 0` and perfectly separated data, no finite minimiser exists, so raising `RuntimeError` is acceptable there.
- Scale: the objective is in the units of `X` as given. The penalty applies to the coefficients of the features as supplied, whatever their scale, and the features may be strongly correlated.
- Large scores: on separable data with `l2 > 0` the minimiser is finite, and the fit must return it. Linear scores may reach the hundreds or thousands during the iteration; no intermediate value may overflow.
- Must not mutate its inputs.

## Examples

```python
>>> b, w = fit_logistic_regression(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1]), l2=1.0)
>>> round(b, 4), np.round(w, 4)
(-0.5748, array([0.3832]))
>>> fit_logistic_regression(np.array([[0.0], [1.0]]), np.array([1, 1]))
ValueError: y must contain both classes; a single class has no finite fit
>>> fit_logistic_regression(np.array([[0.0], [1.0]]), np.array([0, 1]), l2=-0.5)
ValueError: l2 must be a finite number that is zero or positive
```

On the raw `breast_cancer` features (569 rows, 30 columns, column means from about 0.004 to about 880, nearly separable), `fit_logistic_regression(X, y, l2=1.0)` returns a finite fit whose gradient meets `tol`. The same data with a weak penalty (`l2=0.001`) and labels from a linear rule on those features is also perfectly separable; the fit stays finite even though its linear scores are in the thousands.

## Constraints

- Allowed: NumPy and the standard library (`math`). Write the loss, gradient and solver yourself.
- Not allowed: `scipy`, `sklearn`, `statsmodels`, `torch`, `jax` and `autograd`, including their
  optimisers and automatic differentiation. Newton's method, IRLS or gradient-based methods are all fine,
  provided they reach the optimum to `tol`.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: agreement with an independent
L-BFGS-B minimisation of the same objective on random problems; the gradient condition at the returned
point; the unpenalised intercept, both in closed form (all-zero features give the logit of the positive
rate) and as calibration in the large (mean predicted probability equals the positive rate); finite
coefficients on nearly separable data; agreement with the unpenalised fit when `l2 = 0`; identical columns
sharing their coefficient; a `RuntimeError` when the iteration budget is too small; rejection of bad
`l2`, labels, classes, shapes, non-finite values and budgets; and that the inputs are left unchanged.

The realism families covered by the hidden tests are: features on scales spanning many orders of magnitude, strongly correlated real features, nearly separable real labels under a weak penalty, perfectly separable labels whose linear scores pass the hundreds, and a rare positive class (about 1 in 100) on separable real features. Each family is checked against an independent optimiser, the gradient condition or finiteness, and the run must not emit a NumPy warning.
