# Gradient descent

Implement `gradient_descent(grad, x0, lr, max_iters=10_000, tol=1e-8)`. It minimises a
differentiable function `f` without knowing where its minimiser is. Starting from `x0`, repeat
the update `x ← x − lr · grad(x)`, where `grad(x)` returns the gradient of `f` at `x`. Stop as soon
as the gradient is small, that is when ‖grad(x)‖ ≤ `tol` (Euclidean norm), or when `max_iters`
updates have been made. Return the final point and the number of updates applied.

## Contract

- `grad`: a callable that takes a 1-D float array `x` of length d and returns the gradient as an
  array of shape (d,).
- `x0`: 1-D array-like of finite real numbers, length d ≥ 1. Lists are allowed.
- `lr`: the learning rate (step size), a number.
- `max_iters`: int, default 10_000. The maximum number of updates.
- `tol`: float, default 1e-8. The gradient-norm tolerance.
- Returns: `(x, n_iters)`. `x` is a new float64 array of shape (d,). `n_iters` is the number of
  updates applied. If `x0` already meets the tolerance, `n_iters == 0` and `x` equals `x0`. If
  `max_iters` updates are not enough, return the last iterate and `n_iters == max_iters`.
- Raises `ValueError` when `lr` is zero, negative or NaN (this check comes first, so it applies even
  when `x0` is already optimal); when `x0` contains NaN or infinity; and when an iterate or gradient
  becomes non-finite, with the message `diverged`.
- Must not mutate `x0`. The returned `x` must not share memory with `x0`.

## Examples

```python
>>> gradient_descent(lambda x: 2 * (x - 3), [0.0], lr=0.5)
(array([3.]), 1)        # one update lands exactly on the minimum x = 3
>>> gradient_descent(lambda x: 2 * (x - 3), [0.0], lr=0.25)
(array([3.]), 30)       # same minimum, more updates: the error halves each step
>>> gradient_descent(lambda x: 2 * (x - 3), [0.0], lr=1.5)
ValueError: diverged    # each update overshoots by twice the distance, so iterates grow
>>> gradient_descent(lambda x: 2 * (x - 3), [0.0], lr=0.0)
ValueError              # the learning rate must be positive
```

## Constraints

- Allowed: NumPy array arithmetic (`np.array`, `np.asarray`, `np.isfinite`, `np.all`, `np.sqrt`,
  `@`, …).
- Not allowed: `numpy.linalg` and `scipy`. In particular no `np.linalg.solve` and no `np.linalg.norm`,
  and no `scipy.optimize`. The point is the iteration, not a direct solve.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with a direct
linear solve on random well-conditioned quadratics; a step size inside the stability limit
converging, and one past it raising `diverged`; the exact update count on a one-dimensional
quadratic; the cap on `max_iters`; a start point that is already optimal; invalid learning rates,
including when `x0` is already optimal; non-finite and integer start points; and inputs left
unchanged (the start point, and the returned array versus `x0`).
