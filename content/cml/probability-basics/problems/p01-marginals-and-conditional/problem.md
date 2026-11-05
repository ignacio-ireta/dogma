# Marginals and conditional from a joint table

Implement `marginals_and_conditional(joint)`. The input is a joint probability table
$P[i, j] = P(X=i, Y=j)$ for two discrete random variables. Return the two marginals and the
conditional $P(Y=j \mid X=i)$ for every row. Where a row has zero probability, its conditional is
undefined, so return NaN for that row.

## Contract

- `joint`: 2-D array-like of shape `(m, k)`, $m, k \ge 1$. Entries are finite, non-negative, and
  sum to 1 within $10^{-9}$. Integers are allowed.
- Returns: a tuple `(p_x, p_y, p_y_given_x)` of NumPy `float64` arrays:
  - `p_x`, shape `(m,)`: row sums, $P(X=i)$.
  - `p_y`, shape `(k,)`: column sums, $P(Y=j)$.
  - `p_y_given_x`, shape `(m, k)`: $P(Y=j \mid X=i) = P[i,j] / P(X=i)$. A row with $P(X=i) = 0$
    is entirely `NaN`.
- Raises `ValueError` when `joint` is not 2-D, when any entry is negative or non-finite (NaN or
  inf), or when the entries do not sum to 1 within $10^{-9}$. An empty table is rejected this way.
- Must not mutate its input.

## Examples

```python
>>> p_x, p_y, cond = marginals_and_conditional([[0.5, 0.0], [0.0, 0.5]])
>>> p_x, p_y
(array([0.5, 0.5]), array([0.5, 0.5]))
>>> cond                                  # knowing X fixes Y
array([[1., 0.],
       [0., 1.]])
>>> _, _, cond = marginals_and_conditional([[0.0, 0.0], [0.5, 0.5]])
>>> cond[0]                               # P(X=0) = 0, so the row is undefined
array([nan, nan])
>>> marginals_and_conditional([[0.5, 0.5], [0.5, 0.5]])   # sums to 2
ValueError: joint must sum to 1
```

## Constraints

- Allowed: NumPy (`np.array`, `np.isfinite`, `sum`, broadcasting, …).
- Not allowed: `pandas` and `scipy`. Compute the sums with NumPy so that the pandas reference in the
  hidden tests is an independent check.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with an
independent pandas computation on random tables of several shapes, including non-square ones.
They also check the probability identities: each defined conditional row sums to 1, the marginals
each sum to 1, and $P(X=i)\,P(Y=j \mid X=i)$ rebuilds the joint. The families also include
independent tables, zero-probability rows, very small but positive marginals, single-row and
single-column tables, integer tables, sums off by more than the tolerance, negative and NaN entries,
empty and one-dimensional input, and inputs left unchanged.
