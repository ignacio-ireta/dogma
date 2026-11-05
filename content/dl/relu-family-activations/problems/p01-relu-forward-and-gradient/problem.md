# ReLU-family forward and local gradient

Build an elementwise rectifier that returns both its activation and the local derivative needed
by a backward pass. One fixed slope covers ordinary ReLU and leaky ReLU.

## Contract

- `x`: real numeric array-like of any shape, including a scalar or empty array. Convert to `float64`.
- `negative_slope`: real scalar, finite and in `[0, 1]`; default `0` is ReLU.
- For each value `v`, return `v` if `v > 0`, otherwise `negative_slope * v`.
- Return `(values, derivatives)`: two NumPy `float64` arrays, each exactly the shape of `x`.
  Derivative is `1` for positive values, `negative_slope` for negative values, and **0 at zero**.
  The zero convention is fixed even when the slope is 1.
- Raise `ValueError` for nonfinite or complex `x`, or a nonfinite/out-of-range slope.
  Other nonnumeric inputs are outside the contract. Must not mutate inputs.

Returned arrays must not share storage with inputs.

## Examples

```python
>>> relu_family([-2, 0, 3])
(array([-0., 0., 3.]), array([0., 0., 1.]))
```

```python
>>> relu_family([-2, 0, 3], 0.1)
(array([-0.2, 0., 3.]), array([0.1, 0., 1.]))
```

## Constraints

Allowed: NumPy and Python standard library, including math.erf.
Not allowed: torch, jax, scipy, sklearn. Implement the formulas from scratch.

## Done looks like

The examples and edge cases pass. Hidden tests cover a seeded independent oracle, degenerate
inputs, output types, invalid arguments, scale extremes within the contract and input preservation.
Gradient outputs are also checked by finite differences away from nondifferentiable points.
