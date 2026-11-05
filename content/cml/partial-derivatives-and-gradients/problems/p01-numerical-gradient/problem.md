# Numerical gradient

Implement `numerical_gradient(f, x)`, which estimates the gradient $\nabla f(\mathbf{x})$ of a scalar
function $f:\mathbb{R}^n\to\mathbb{R}$ at a point $\mathbf{x}$. The $i$-th component is the partial
derivative $\partial f/\partial x_i$: vary coordinate $i$ by a small step, hold every other coordinate
fixed, and take a symmetric difference. Use one such difference per coordinate, so that each
perturbed point changes exactly one coordinate.

## Contract

- `f`: a callable that takes a 1-D float64 array of shape `(n,)` and returns a real number. Only
  evaluate it.
- `x`: a 1-D array-like of finite real numbers, length $n \geq 1$. Integers are allowed.
- Returns: a NumPy array of shape `(n,)` and dtype `float64`. Component $i$ is
  $\dfrac{f(\mathbf{x} + h_i\mathbf{e}_i) - f(\mathbf{x} - h_i\mathbf{e}_i)}{2h_i}$ with
  $h_i = \varepsilon^{1/3}\max(1, |x_i|)$, where $\varepsilon$ is the float64 machine epsilon.
- Raises `ValueError` when `x` is not 1-D, is empty, or contains a NaN or infinite entry.
- Must not mutate `x`.

## Examples

```python
>>> np.round(numerical_gradient(lambda v: float(np.sum(v**2)), [1.0, 2.0]), 6)
array([2., 4.])
>>> np.round(numerical_gradient(lambda v: 3 * v[0] - v[1], [0.7, -2.0]), 6)
array([ 3., -1.])
>>> numerical_gradient(lambda v: 0.0, [])
ValueError: x must be a non-empty 1-D array
```

## Constraints

- Allowed: NumPy (`np.asarray`, `np.empty`, `np.isfinite`, ...) and the `math` module.
- Not allowed: `numpy.gradient`, `scipy.optimize.approx_fprime` and `scipy.misc`. These are
  library gradient routines, so write the difference quotients yourself.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: a quadratic form against
its closed-form gradient $(A + A^\top)\mathbf{x}$ on random matrices; a log-sum-exp function against
the softmax; the Rosenbrock function against its closed form; every partial against a complex-step
reference on a composite function; a one-dimensional square at two points, where a one-sided step
would be visibly off; very large coordinates, where the step must scale with $|x_i|$; an integer
input; empty, two-dimensional and non-finite input; and that `x` is left unchanged.
