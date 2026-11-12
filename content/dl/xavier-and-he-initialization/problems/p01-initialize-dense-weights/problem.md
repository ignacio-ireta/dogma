# Initialize a dense weight matrix

Implement two initialization policies for a dense layer whose weights are stored in
`(fan_in, fan_out)` order. Sampling and scale are part of the contract, so independent
implementations can reproduce the same initial model from a seed.

## Contract

- `fan_in`, `fan_out`: positive Python integers, excluding bool and NumPy integer scalars.
- `scheme`: exactly `"xavier_uniform"` or `"he_normal"`.
- `seed`: nonnegative Python integer, excluding bool. Use a fresh `np.random.default_rng(seed)`.
- Return a `float64` NumPy array of shape `(fan_in, fan_out)`.
- For Xavier, set `a = sqrt(6 / (fan_in + fan_out))` and draw with
  `rng.uniform(-a, a, size=(fan_in, fan_out))`.
- For He, set `s = sqrt(2 / fan_in)` and draw with
  `rng.normal(0, s, size=(fan_in, fan_out))`.
- Raise `ValueError` for invalid fans, scheme or seed. Keep the global NumPy RNG state
  unchanged. Calls must not share RNG state or output storage. Matrix sizes must be feasible
  in memory, with fans representable as NumPy dimensions; allocation failures are out of scope.
- This initializes weights only. Biases are a separate policy and are not returned.

## Examples

```python
>>> np.round(initialize_dense(2, 3, "xavier_uniform", 7), 6)
array([[ 0.27407 ,  0.870252,  0.603997],
       [-0.602041, -0.437814,  0.818415]])
```

```python
>>> np.round(initialize_dense(1, 1, "he_normal", 0), 6)
array([[0.177809]])
```

## Constraints

Allowed: NumPy and Python standard library, including math.erf.
Not allowed: torch, jax, scipy, sklearn. Implement the formulas from scratch.

## Done looks like

The examples and edge cases pass. Hidden tests cover a seeded independent oracle, degenerate
inputs, output types, invalid arguments, scale extremes within the contract and input preservation.

Some hidden tests take their inputs from the real datasets `breast_cancer`, `diabetes`, `digits`,
`iris` and `wine` (see `docs/DATASETS.md`).
