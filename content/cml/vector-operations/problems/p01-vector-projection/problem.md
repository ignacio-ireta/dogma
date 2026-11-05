# Vector projection

Implement `vector_projection(a, b)`, which returns the orthogonal projection of vector `a` onto
vector `b`: the point on the line through `b` that is closest to `a`.

## Contract

- `a`, `b`: 1-D sequences of real numbers (Python lists or NumPy arrays) of the same length
  $n \ge 1$. Integers are allowed.
- Returns: a NumPy array of shape `(n,)` and dtype `float64`, equal to
  $\dfrac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}\,\mathbf{b}$.
- Raises `ValueError` when `a` or `b` is not 1-D, when their lengths differ, when they are empty,
  or when every entry of `b` is zero. A nonzero `b` with tiny or huge entries is valid input.
- Must not mutate its inputs.

## Examples

```python
>>> vector_projection([3, 4], [1, 0])
array([3., 0.])
>>> vector_projection([1, 2, 3], [1, 1, 1])     # the mean, repeated
array([2., 2., 2.])
>>> vector_projection([1, 0], [0, 5])           # orthogonal → zero vector
array([0., 0.])
>>> vector_projection([1, 2], [0, 0])
ValueError: cannot project onto the zero vector
```

## Constraints

- Allowed: NumPy (`@`, `np.sum`, `np.asarray`, …).
- Not allowed: `numpy.linalg`. Compute dot products yourself.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with an
independent reference on random vectors, the geometric properties of a projection (residual
orthogonal to `b`, idempotence, invariance to rescaling `b`), invalid shapes, integer input,
very large and very small magnitudes (including a nonzero `b` whose squared length would overflow
or underflow), and that inputs are left unchanged.
