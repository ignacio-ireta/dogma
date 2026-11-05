# Gradient boosting with stumps for squared error

Boost depth-one regression trees (stumps) on the squared error. The model starts at the mean of the targets. Each round fits a stump to the current residuals, the negative gradient of the squared error, and adds a shrunken copy of it. Implement `boost_stumps_mse(X, y, *, n_estimators, learning_rate)`, which returns a `predict` function and the training predictions.

### The procedure
- $F_0(\mathbf x)=\bar y$ for every row.
- For $m=1,\dots,M$ (`n_estimators`): the residuals are $r_i=y_i-F_{m-1}(\mathbf x_i)$. Fit a stump $h_m$ to them:
  - For every feature $j$ in increasing order, and every threshold $t$ between consecutive distinct sorted values of $X_{:,j}$, in increasing order, let the left child be $\{X_{ij}\le t\}$ and the right child $\{X_{ij}>t\}$. Each child predicts the mean of the residuals in it.
  - The gain is the reduction in sum of squared errors, $\mathrm{SSE}(\text{parent})-\mathrm{SSE}(\text{left})-\mathrm{SSE}(\text{right})$, where each SSE is taken around its own mean.
  - Scan the candidates in that order. A later candidate replaces the incumbent only when its gain exceeds the incumbent's by more than $10^{-9}$. So a tie goes to the lowest feature index, then the smallest threshold.
  - If no feature has two distinct values, $h_m$ is the constant mean of the residuals.
  - Update: $F_m=F_{m-1}+\eta\,h_m$, where $\eta$ is `learning_rate`.

## Contract
- `X`: 2-D array-like of finite numbers, shape `(n, d)`, with $n\ge1$ and $d\ge1$.
- `y`: 1-D array-like of finite numbers, length `n`.
- `n_estimators`: keyword-only integer, at least 1. `learning_rate`: keyword-only finite number greater than 0.
- Returns: `(predict, train_pred)`.
  - `predict(X_new)` takes a 2-D array with `d` columns and returns a `float64` array of shape `(m,)`, equal to $F_M$ at each row.
  - `train_pred` is a `float64` array of shape `(n,)`, equal to $F_M(X)$.
- Raises: `ValueError` for bad shapes, non-finite values, a non-integer or too-small `n_estimators`, a `learning_rate` that is not finite and positive, and, inside `predict`, the wrong number of columns or non-finite values.
- Feature values may be as large as `1e308`. Compute each midpoint without overflow, so that a stump still separates the two values.
- Must not mutate `X` or `y`.

## Examples

```python
>>> import numpy as np
>>> X = np.array([[1.0], [2.0], [3.0], [4.0]])
>>> y = np.array([0.0, 0.0, 1.0, 1.0])
>>> predict, train_pred = boost_stumps_mse(X, y, n_estimators=1, learning_rate=1.0)
>>> train_pred.tolist()
[0.0, 0.0, 1.0, 1.0]
>>> predict(np.array([[2.5]])).tolist()
[0.0]
>>> _, half = boost_stumps_mse(X, y, n_estimators=1, learning_rate=0.5)
>>> half.tolist()
[0.25, 0.25, 0.75, 0.75]
```

The first round starts at $\bar y=0.5$, and the best stump splits at $t=2.5$, the midpoint of 2 and 3, with leaves $-0.5$ and $+0.5$. Because $x\le t$ goes left, $x=2.5$ falls in the left leaf and predicts $0.5-0.5=0.0$.

## Constraints

- Allowed: NumPy and the standard library (`math`).
- Not allowed: `sklearn` (including `GradientBoostingRegressor`) and `scipy`. Write the stump search yourself.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: agreement with a brute-force implementation of the same procedure on random data, at training points and at new points; agreement with scikit-learn's gradient boosting with depth-one trees on tie-free data; that `predict` reproduces the returned training predictions; constant features; ties broken by feature index and by threshold; the effect of the learning rate; a training error that does not increase with more rounds; a single row; invalid inputs and arguments; and inputs left unchanged.
