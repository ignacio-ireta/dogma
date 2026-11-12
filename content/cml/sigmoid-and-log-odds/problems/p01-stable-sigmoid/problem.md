# Stable sigmoid

Implement `sigmoid(z)`, the logistic function $\sigma(z) = \dfrac{1}{1 + e^{-z}}$, applied elementwise.
It maps any real logit to a probability in $[0, 1]$. The naive formula overflows for large negative
$z$, and a two-branch formula can still emit overflow warnings from the branch it does not use.
Build the function so that every finite input, including logits with $|z|$ up to $10^4$, gives a
finite answer and no NumPy warning.

## Contract

- `z`: array-like of real numbers, any shape. A scalar, an empty array and `+inf` / `-inf` are valid.
  The limits are $\sigma(+\infty) = 1$ and $\sigma(-\infty) = 0$.
- Returns: `float64` NumPy array with the same shape as `z` (a 0-d array for a scalar). Every value lies in $[0, 1]$.
- Accuracy: $\sigma(0) = 0.5$ exactly. For $z < 0$ the result keeps its relative precision, so a tiny
  value such as $\sigma(-36)$ is not rounded to zero or to a value that is off by a large factor.
- Raises: `ValueError` when `z` contains NaN.
- Must not mutate its input.
- Must not emit a NumPy overflow, invalid-value or divide warning for any non-NaN input.

## Examples

```python
>>> sigmoid(0.0)
array(0.5)
>>> sigmoid(np.array([-2.0, 0.0, 2.0]))
array([0.11920292, 0.5       , 0.88079708])
>>> sigmoid(np.array([-1e4, 1e4]))          # saturates to exact 0 and 1, with no warning
array([0., 1.])
>>> sigmoid(np.array([[1.0, -1.0], [0.0, 3.0]])).shape
(2, 2)
```

## Constraints

- Allowed: NumPy and the standard library. Write the stable formula yourself.
- Not allowed: `scipy`, including `scipy.special.expit`. The stable sigmoid is the point of this exercise.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: agreement with an
independent stable reference on random inputs; finite values in $[0, 1]$ with no warnings for
logits of very large magnitude, in both directions; relative precision on the negative tail, against
a 60-digit reference; exact $0.5$ at zero and the symmetry $\sigma(-z) = 1 - \sigma(z)$ in the bulk;
the infinite limits; NaN input; empty, scalar and matrix shapes; integer input; and that the input is
left unchanged.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).
