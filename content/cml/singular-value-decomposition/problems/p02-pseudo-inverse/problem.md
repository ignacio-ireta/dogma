# Pseudo-inverse

Implement `pseudo_inverse(A, *, rcond=1e-15)`, the Moore-Penrose pseudo-inverse of a real matrix, computed
from its singular value decomposition. Singular values at or below a cutoff are treated as zero, which
makes the function work for rectangular and rank-deficient matrices. `rcond` is keyword-only. Use
`numpy.linalg.svd`.

## Contract

- `A`: 2-D array-like of real numbers, shape `(m, n)` with `m >= 1` and `n >= 1`. Integers allowed. Every
  entry is finite.
- `rcond`: keyword-only, a finite real number `>= 0`, default `1e-15`. Booleans are rejected.
- Cutoff: let `s_max` be the largest singular value of `A`. Every singular value `s_i <= rcond * s_max` is
  treated as zero. The others are inverted. This is the rule of `numpy.linalg.pinv(A, rcond=rcond)`. The
  cutoff is relative to `s_max`, not absolute. If `A` is the zero matrix, every singular value is zero and
  the result is the zero matrix.
- Returns: a float64 array of shape `(n, m)`, equal to `V diag(s_plus) U^T`, where `s_plus_i = 1 / s_i` for
  kept singular values and `0` otherwise.
- Raises `ValueError` when `A` is not 2-D, is empty, contains NaN or infinity, or when `rcond` is not a
  finite number `>= 0`.
- Must not mutate its input.

## Examples

```python
>>> pseudo_inverse([[2.0, 0.0], [0.0, 4.0]])          # up to rounding
array([[0.5 , 0.  ],
       [0.  , 0.25]])
>>> pseudo_inverse([[1.0, 2.0], [2.0, 4.0]])          # rank one: the zero singular value is dropped
array([[0.04, 0.08],
       [0.08, 0.16]])
>>> pseudo_inverse([[1000.0, 0.0], [0.0, 0.5]], rcond=1e-3)   # cutoff = 1e-3 * 1000 = 1, so 0.5 is dropped
array([[0.001, 0.   ],
       [0.   , 0.   ]])
>>> pseudo_inverse([[1.0, 0.0]], rcond=-1.0)          # negative cutoff
ValueError
```

## Constraints

- Allowed: NumPy array operations, including `numpy.linalg.svd`.
- Not allowed: `numpy.linalg.pinv`, `numpy.linalg.lstsq` and `scipy`. Build the inverse from the SVD.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with NumPy's
pseudo-inverse on random tall, wide and square matrices, and on rank-deficient matrices under an
explicit cutoff; the four Penrose identities on random rank-deficient matrices; the default cutoff,
dropping a singular value below it as well as keeping one above it; the cutoff being relative to the
largest singular value, including very small and very large scales; a singular value equal to the
cutoff, which is dropped; the rank-one closed form; the zero matrix; invalid input and `rcond`;
integer input; scaling by a constant; and inputs left unchanged.
