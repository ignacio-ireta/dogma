# Vector p-norm

Implement `vector_norm(x, p)`, which returns the $p$-norm of a one-dimensional vector:
$\lVert\mathbf{x}\rVert_p = \big(\sum_i |x_i|^p\big)^{1/p}$ for a real $p \geq 1$, and
$\lVert\mathbf{x}\rVert_\infty = \max_i |x_i|$ when `p` is infinity. The result must be correct for
very large and very small entries. Squaring the entries is the usual place where this goes wrong,
so do not let the intermediate values overflow or underflow.

## Contract

- `x`: a one-dimensional sequence (list or NumPy array) of finite real numbers, length $n \geq 1$.
  Integers are allowed.
- `p`: a real number with $p \geq 1$, or infinity (`math.inf` or `np.inf`).
- Returns: a Python `float` that is at least 0. The zero vector has norm 0 for every `p`.
- Raises `ValueError` when:
  - `p` is below 1, or is NaN. Exponents below 1 do not define a norm.
  - `x` is not one-dimensional, or is empty.
- Must not mutate its input.
- Entries anywhere in the float64 range must give the correct finite result when the norm itself
  is representable. For example, `[3e200, 4e200]` has L2 norm $5\cdot10^{200}$ and `[3e-200, 4e-200]`
  has L2 norm $5\cdot10^{-200}$.

## Examples

```python
>>> vector_norm([3, 4], 2)
5.0
>>> vector_norm([3, -4], 1)
7.0
>>> vector_norm([3, -4], float("inf"))
4.0
>>> vector_norm([3e200, 4e200], 2)   # naive squaring overflows; the exact answer is 5e200
4.9999999999999995e+200                # 5e200, up to float64 rounding
>>> vector_norm([1, 2], 0.5)
ValueError: p must be at least 1 (or infinity); got 0.5
```

## Constraints

- Allowed: NumPy (`np.abs`, `np.max`, `np.sum`, `**`), and the `math` module.
- Not allowed: `numpy.linalg`. Compute the norm yourself, so that you control the scaling.

## Done looks like

All public tests pass. Then **Submit** runs the hidden tests. They compare against NumPy's norm on
random vectors for several values of $p$ (1, 2, a non-integer, and infinity), in the moderate range
where NumPy is reliable. They check the norm axioms: homogeneity, the triangle inequality, and the
ordering of norms as $p$ grows. They cover the zero vector, single entries, invalid `p` (below 1, or
NaN), non-one-dimensional and empty input, inputs left unchanged, and extreme magnitudes for the L2
norm and for a non-integer $p$, both very large and very small.
