# Regression report

Implement `regression_report(y_true, y_pred)`. It returns the four standard scores for one set of
predictions: the mean absolute error, the mean squared error, the root mean squared error and the
coefficient of determination R². Compute each one from its formula with NumPy. The library
implementations are the oracle, not the solution.

## Contract

- `y_true`: 1-D array-like of finite real numbers, length n ≥ 1.
- `y_pred`: 1-D array-like of finite real numbers, the same length n.
- Returns: a `dict` with exactly the keys `"mae"`, `"mse"`, `"rmse"` and `"r2"`, each a Python float.
  - `mae` = (1/n) Σ |yᵢ − ŷᵢ|
  - `mse` = (1/n) Σ (yᵢ − ŷᵢ)²
  - `rmse` = √`mse`
  - `r2` = 1 − Σ (yᵢ − ŷᵢ)² / Σ (yᵢ − ȳ)², where ȳ is the mean of `y_true`. `r2` is not clipped, so it can be negative.
- Raises `ValueError` when an input is not 1-D; when the lengths differ; when the input is empty;
  when any value is NaN or infinite; or when every `y_true` value is equal. In that last case the
  denominator of R² is zero and R² is undefined.
- Compute the sums of squares without underflow. `y_true = [-1e-200, 0, 1e-200]` is not constant, so its
  `r2` is defined even though the squares underflow.
- Must not mutate inputs.

## Examples

```python
>>> regression_report([3.0, -0.5, 2.0, 7.0], [2.5, 0.0, 2.0, 8.0])
{'mae': 0.5, 'mse': 0.375, 'rmse': 0.6123724356957945, 'r2': 0.9486081370449679}
>>> regression_report([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
{'mae': 2.0, 'mse': 5.0, 'rmse': 2.23606797749979, 'r2': -3.0}
>>> regression_report([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])
ValueError   # every y_true value is 2.0, so R² is undefined
```

## Constraints

- Allowed: NumPy.
- Not allowed: `sklearn` and `scipy`. Compute every score from its formula.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare each score with
scikit-learn's metric on random data, and check that R² is not clipped for very poor predictions. They
also check the zero-variance rule, the mean predictor (R² = 0), perfect predictions, and how each score
changes under scaling and under a common shift of both vectors. They check the rejected inputs (shape,
length, empty, non-finite, constant truth), that integer input gives the same scores as float input,
that inputs are left unchanged, and that every score is a scalar.

The hidden tests draw their bulk inputs from the real dataset `california_housing` (see
`docs/DATASETS.md`).
