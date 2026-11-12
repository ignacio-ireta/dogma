# Logit (log-odds)

Implement `logit(p)`, the inverse of the logistic function: $\operatorname{logit}(p) = \log\dfrac{p}{1 - p}$.
It turns a probability into its log-odds, the quantity a logistic regression is linear in. Probabilities
close to $0$ or $1$ make $p$ and $1 - p$ lose digits, and probabilities close to $\tfrac12$ make the
ratio $p/(1-p)$ lose relative precision in the result, because $\operatorname{logit}(p)$ is then close to $0$.
Compute it so that the result keeps its relative precision across the whole open interval.

## Contract

- `p`: array-like of real numbers, any shape. Every entry must be finite and strictly between 0 and 1.
- Returns: `float64` NumPy array with the same shape as `p` (a 0-d array for a scalar).
- Raises: `ValueError` when any entry is NaN, infinite, $\le 0$ or $\ge 1$, including $0$ and $1$ themselves.
- Accuracy: $\operatorname{logit}(\tfrac12) = 0$ exactly. The result is odd about one half,
  $\operatorname{logit}(1 - p) = -\operatorname{logit}(p)$.
- Must not mutate its input. Must not emit a NumPy warning for any valid input.

## Examples

```python
>>> logit(0.5)
array(0.)
>>> logit(np.array([0.25, 0.75]))
array([-1.09861229,  1.09861229])
>>> logit(np.array([[0.2, 0.8], [0.5, 0.9]])).shape
(2, 2)
>>> logit(1.0)                     # 1 is not strictly inside (0, 1)
ValueError: logit is defined only for probabilities strictly between 0 and 1
```

## Constraints

- Allowed: NumPy and the standard library.
- Not allowed: `scipy`, including `scipy.special.logit`. Deriving a numerically accurate form is the exercise.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: agreement with an
independent reference on random probabilities; relative precision near $\tfrac12$, near $0$ and near
$1$, each against a 60-digit reference; the odd symmetry about one half; the round trip through the
logistic function; the boundary and out-of-range rejections (zero, one, negative, above one, NaN and
infinity); empty and matrix shapes; float32 input; and that the input is left unchanged.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).
