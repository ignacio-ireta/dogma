# Best rank-k approximation

Implement `best_rank_k(A, k)`, which returns the rank-`k` matrix closest to `A` in the Frobenius norm,
together with the size of the error. This is the truncated singular value decomposition: keep the `k`
largest singular values and their singular vectors, and drop the rest. Use `numpy.linalg.svd`.

## Contract

- `A`: 2-D array-like of real numbers, shape `(m, n)` with `m >= 1` and `n >= 1`. Integers allowed.
  Every entry is finite.
- `k`: an `int` with `0 <= k <= min(m, n)`. Booleans are rejected.
- Returns: a tuple `(A_k, error)`.
  - `A_k`: float64 array of shape `(m, n)`. If the singular values of `A` are `s_1 >= s_2 >= ...`, with
    singular vectors `u_i` and `v_i`, then `A_k = sum of s_i u_i v_i^T` over `i <= k`. It is the zero
    matrix when `k = 0`.
  - `error`: a Python float, `sqrt(s_{k+1}^2 + s_{k+2}^2 + ...)`, which is the Frobenius norm of
    `A - A_k`. When `k` is at least the rank of `A`, the error is zero up to rounding.
- When `s_k = s_{k+1}`, several rank-`k` matrices are optimal and any one of them is accepted. The
  error is the same for all of them.
- Raises `ValueError` when `A` is not 2-D, is empty, contains NaN or infinity, or when `k` is not an
  integer in `[0, min(m, n)]`.
- Must not mutate its input.

## Examples

```python
>>> A_k, err = best_rank_k([[3.0, 0.0], [0.0, 1.0]], 1)
>>> A_k
array([[3., 0.],
       [0., 0.]])
>>> err
1.0
>>> best_rank_k([[1.0, 2.0], [3.0, 4.0]], 0)[0]     # k = 0 keeps nothing
array([[0., 0.],
       [0., 0.]])
>>> best_rank_k([[1.0, 2.0], [3.0, 4.0]], 3)        # k above min(m, n) = 2
ValueError
```

## Constraints

- Allowed: NumPy array operations, including `numpy.linalg.svd`.
- Not allowed: `scipy`.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with an independent
eigen-decomposition of `A^T A` on random matrices with distinct singular values; that the error equals
the Frobenius norm of the residual and the square root of the discarded squared singular values; that no
random rank-`k` matrix does better; the rank of `A_k`; the boundary values `k = 0` and `k = min(m, n)`;
the zero matrix; a rank-one input, where the error must vanish; repeated singular values (invariants
only); wide, tall and square shapes; large and small magnitudes; integer input; invalid input; and inputs
left unchanged.
