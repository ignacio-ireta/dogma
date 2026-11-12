# Dense layer vector-Jacobian product

Implement `dense_backward(X, W, G, reduction="sum")` for the affine map
`Y = X @ W + b`. Return the input, weight, and broadcast-bias gradients,
without constructing the full Jacobian. `G` is an unreduced upstream signal.

## Contract

- `X`, `W`, `G`: real numeric NumPy arrays, shapes `(n,d)`, `(d,k)`, `(n,k)`.
  Every dimension is 1..64 and every entry is finite with absolute value <=1e6.
  Integer and strided arrays are valid; complex, bool, and object arrays are invalid.
- `reduction`: exactly `"sum"` or `"mean"`. For sum, differentiate
  `sum((X @ W + b) * G)`. For mean, differentiate that scalar divided by `n`.
  This applies the factor to **all three** gradients. The bias value is irrelevant.
- Return `(dX, dW, db)`, float64 arrays of shapes `(n,d)`, `(d,k)`, `(k,)`.
- Raise `ValueError` for invalid ranks, shapes, bounds, dtypes, nonfinite entries,
  or reduction. Do not mutate input arrays.

## Examples

```python
dense_backward(np.array([[2.0, 3.0]]), np.array([[4.0], [5.0]]), np.array([[2.0]]))
# -> (array([[8., 10.]]), array([[4.], [6.]]), array([2.]))
```

```python
dense_backward(np.array([[1.0], [3.0]]), np.array([[2.0]]), np.ones((2, 1)), "mean")
# -> (array([[1.], [1.]]), array([[2.]]), array([1.]))
```

## Constraints

Allowed: Python stdlib and NumPy, including matrix multiplication. Not allowed:
`torch`, `jax`, `tensorflow`, or `sklearn`. Write the backward map yourself.
The default tolerance is atol=1e-8, rtol=1e-6; finite differences use fixed
small perturbations of smooth scalar objectives.

## Done looks like

Public examples and empty-input rejection pass. Hidden families cover independent
scalar-loop comparisons, finite differences for X/W/b under both reductions,
non-square and strided shapes, integer casting, scale extremes, zero upstream,
invalid inputs, and mutation. Use sum when an external loss already averaged G.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `digits` (see
`docs/DATASETS.md`).
