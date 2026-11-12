# Constrained quadratic minimum

Implement `constrained_quadratic_min(Q, c, A, b)`, which minimises the quadratic objective
$f(\mathbf{x}) = \tfrac12\mathbf{x}^\top Q\mathbf{x} + \mathbf{c}^\top\mathbf{x}$ subject to the linear
equality constraints $A\mathbf{x} = \mathbf{b}$. Solve the KKT system for the minimiser and the
multipliers together. Use the Lagrangian $\mathcal{L}(\mathbf{x},\boldsymbol\lambda) =
f(\mathbf{x}) - \boldsymbol\lambda^\top(A\mathbf{x} - \mathbf{b})$. With this sign, $\boldsymbol\lambda$
is the derivative of the optimal value with respect to $\mathbf{b}$.

## Contract
- `Q`: $n \times n$ real symmetric positive definite matrix, $n \ge 1$. Symmetric up to rounding.
- `c`: 1-D, length $n$.
- `A`: 2-D, shape $(m, n)$ with $m \ge 1$ and full row rank (the rows are linearly independent).
- `b`: 1-D, length $m$.
- Returns: a tuple `(x, lam)`. `x` is a float64 array of shape `(n,)` and `lam` is a float64 array
  of shape `(m,)`. They satisfy $Q\mathbf{x} + \mathbf{c} = A^\top\boldsymbol\lambda$ and $A\mathbf{x} = \mathbf{b}$.
- Raises `ValueError` when: a shape disagrees with the rules above; an entry is not finite; `Q` is
  not symmetric; `Q` is not positive definite; or `A` is rank deficient.
- Must not mutate inputs.

## Examples

```python
>>> x, lam = constrained_quadratic_min(2 * np.eye(2), [0, 0], [[1, 1]], [1])
>>> x, lam
(array([0.5, 0.5]), array([1.]))

>>> x, lam = constrained_quadratic_min(np.eye(2), [-3, 0], [[1, 1]], [1])
>>> x, lam
(array([ 2., -1.]), array([-1.]))

>>> constrained_quadratic_min(np.eye(3), [0, 0, 0],
...                           [[0.1, 0.7, 0.3], [0.3, 2.1, 0.9]], [1, 3])
ValueError: A is rank deficient: its rows are dependent
```

## Constraints
- Allowed: NumPy, including `np.linalg` (for example `solve`, `cholesky`, `matrix_rank`).
- Not allowed: `scipy` (including `scipy.optimize`). Write the KKT system yourself.

## Done looks like
All public tests pass, then **Submit** runs the hidden tests. They check: agreement with an
independent null-space construction on random problems, for both the minimiser and the
multipliers; the multipliers against a finite-difference derivative of the optimal value with
respect to `b`; feasibility and stationarity on random inputs; that dependent constraints and an
indefinite `Q` raise `ValueError` even when a plain solve would not fail; scaling of the objective
and of the constraint rows, including tiny and huge magnitudes; a one-variable case with a hand
answer; invalid shapes, non-finite values and asymmetric matrices; integer input; and that the
inputs are left unchanged.

The hidden tests draw their bulk inputs from the real dataset `wine` (see `docs/DATASETS.md`).
