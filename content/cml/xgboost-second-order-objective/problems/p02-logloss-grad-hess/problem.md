# Log-loss gradient and Hessian

Boosting needs, for every training example, the first and second derivative of the loss with respect to the current raw score (the margin $m_i$, before the sigmoid). For the binary log-loss of one example with label $y_i \in \{0, 1\}$ and margin $m_i$, the derivatives are

$$g_i = \sigma(m_i) - y_i, \qquad h_i = \sigma(m_i)\big(1 - \sigma(m_i)\big), \qquad \sigma(m) = \frac{1}{1 + e^{-m}}.$$

Implement `logloss_grad_hess(y, margin)`, which returns the pair of arrays $(g, h)$ in input order.

The naive formula overflows for large margins. The function must be stable for every finite margin with $|m| \le 10^4$: no NaN, no overflow warning, and a Hessian that is exactly zero (to within tolerance) once the margin saturates.

## Contract
- `y`: 1-D array-like, the same length as `margin`. Every entry is exactly `0` or `1`. Booleans are accepted.
- `margin`: 1-D array-like of finite floats. Magnitudes up to $10^4$ are valid input.
- Returns: `(grad, hess)`, two `float64` arrays of shape `(n,)`.
- Raises: `ValueError` when `y` and `margin` are not 1-D of the same length, when `y` has any entry other than 0 or 1, or when `margin` has a non-finite entry.
- Empty input (length 0) returns two empty arrays.
- Must not mutate inputs.

## Examples

```python
>>> g, h = logloss_grad_hess([1, 0], [0.0, 0.0])
>>> g, h
(array([-0.5,  0.5]), array([0.25, 0.25]))
>>> g, h = logloss_grad_hess([0, 1], [10000.0, -10000.0])
>>> g, h
(array([ 1., -1.]), array([0., 0.]))
>>> logloss_grad_hess([-1, 1], [0.0, 0.0])
ValueError: y must contain only 0 and 1
```

## Constraints
- Allowed: NumPy and the standard library (`math`).
- Not allowed: `scipy` (including `scipy.special.expit`), `sklearn`, `xgboost`, `statsmodels`, `torch`. The stability is the point of the problem, so a library sigmoid is not an answer.

## Done looks like
All public tests pass, then **Submit** runs the hidden tests. They check: the gradient and Hessian against central finite differences of the per-example log-loss over a grid of margins; the closed forms at moderate and random margins; warning-free, finite results at margins near $\pm 10^4$, with an exactly zero Hessian at saturation; the value at margin zero; the output bounds $|g| \le 1$ and $0 \le h \le 1/4$; empty input; rejection of labels other than 0 or 1, length mismatches, non-1-D arrays and non-finite margins; dtype and shape of the outputs; and that the inputs are left unchanged.
