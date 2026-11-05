# Seeded inverted dropout

Implement `dropout(X, G, p, training, seed)`, returning the forward output,
the gradient with respect to input X, and the exact sampled multiplier.
The combined operation makes forward/backward mask reuse explicit.

## Contract

- `X`, `G`: real numeric NumPy arrays of the same nonempty rank-2 shape `(n,d)`.
  Dimensions are 1..64; entries are finite with absolute value <=1e6. Integer
  and strided arrays are valid; complex, bool, and object arrays are invalid.
- `p`: finite Python int/float in `[0,0.95]`, interpreted as drop probability.
  `training`: Python bool. `seed`: non-boolean integer in `[0,2**32-1]`.
- Training: draw exactly `U=np.random.default_rng(seed).random(X.shape)` and
  set `M=(U>=p).astype(float)/(1-p)`. Return `(X*M, G*M, M)`.
- Evaluation: M is all ones. Return independent copies of X and G plus M,
  even when p is nonzero. The seed is validated but has no effect on values.
- Returns: three float64 arrays, each the same shape as X. No input mutation
  or use of the global RNG. Each call recreates the seed, so repeated calls
  share a mask. Do not enforce an exact count of retained units.
- Raise `ValueError` for invalid inputs, shapes, dtypes, bounds, mode or seed.

## Examples

```python
dropout(np.array([[2.0, 4.0]]), np.ones((1, 2)), 0.5, True, 0)
# -> (array([[4.,0.]]), array([[2.,0.]]), array([[2.,0.]]))
```

```python
dropout(np.array([[2.0, 4.0]]), np.array([[3.0, 5.0]]), 0.5, False, 0)
# -> (array([[2.,4.]]), array([[3.,5.]]), array([[1.,1.]]))
```

## Constraints

Allowed: Python stdlib and NumPy. Not allowed: `torch`, `jax`, `tensorflow`, or
`sklearn`. Do not resample for backward. Test tolerance is atol=1e-8, rtol=1e-6,
including central differences with the seed held fixed.

## Done looks like

Both examples and p=0 identity pass. Hidden families cover seed-mask oracles,
fixed-mask finite differences, eval identity and independent copies, repeated
seeds, strided and integer arrays, large and tiny entries, all-dropped samples,
input/global-RNG preservation, and invalid inputs. Reproducibility means an
explicit local generator; it does not mean resetting global random state.
