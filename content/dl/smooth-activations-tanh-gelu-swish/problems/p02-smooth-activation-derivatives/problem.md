# Smooth activation local derivatives

Implement the local derivatives of the three activations from p01. These are derivatives with
respect to each element of `x`, with no upstream loss gradient applied.

## Contract

- `x`: real numeric array-like, any shape; scalar and empty arrays accepted. Values must be
  finite and in `[-1000, 1000]`. Complex arrays are invalid.
- Return `(d_tanh, d_gelu, d_swish)`: three NumPy `float64` arrays of the input shape.
- Use `1 - tanh(x)**2`, `Phi(x) + x * phi(x)`, and `s + x * s * (1 - s)`, where
  `s = sigmoid(x)`, `Phi` is the standard normal CDF, and
  `phi(x) = exp(-x*x/2) / sqrt(2*pi)`. GELU is the exact CDF version; Swish has beta 1.
- At zero these derivatives are respectively `1`, `0.5`, `0.5`.
- Raise `ValueError` for complex, nonfinite or out-of-range values. Other nonnumeric values
  are outside the contract. No default floating point warnings; do not mutate inputs.

Returned arrays must not share storage with inputs.

## Examples

```python
>>> smooth_derivatives([0])
(array([1.]), array([0.5]), array([0.5]))
```

```python
>>> tuple(np.round(a, 6) for a in smooth_derivatives([1.0]))
(array([0.419974]), array([1.083315]), array([0.927671]))
```

## Constraints

Allowed: NumPy and Python standard library, including math.erf.
Not allowed: torch, jax, scipy, sklearn. Implement the formulas from scratch.

## Done looks like

The examples and edge cases pass. Hidden tests cover a seeded independent oracle, degenerate
inputs, output types, invalid arguments, scale extremes within the contract and input preservation.
Gradient outputs are also checked by finite differences away from nondifferentiable points.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).
