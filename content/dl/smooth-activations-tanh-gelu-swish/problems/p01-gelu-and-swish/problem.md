# Smooth activation values

Implement elementwise tanh, exact Gaussian error linear unit (GELU), and Swish with fixed
beta 1. They share the idea of smoothly controlling how much of an input passes forward.

## Contract

- `x`: real numeric array-like of any shape, including scalar and empty inputs. Every value
  must be finite and in `[-1000, 1000]`. Complex arrays are invalid.
- Return a tuple of three NumPy `float64` arrays in the order `(tanh, gelu, swish)`, each
  exactly the shape of `x`. The formulas are `tanh(x)`, `x * Phi(x)` and `x * sigmoid(x)`.
  `Phi(x) = (1 + erf(x / sqrt(2))) / 2` is the standard normal CDF. Use this **exact GELU**
  definition rather than the tanh approximation.
- Compute sigmoid without overflowing exponentials at the stated tails. Floating point
  rounding to the limiting value is allowed. Default NumPy floating point warnings must not fire.
- Raise `ValueError` for complex, nonfinite or out-of-range values. Other nonnumeric inputs
  are outside the contract. Do not mutate inputs.

Returned arrays must not share storage with inputs.

## Examples

```python
>>> smooth_activations([0.0])
(array([0.]), array([0.]), array([0.]))
```

```python
>>> tuple(np.round(a, 6) for a in smooth_activations([-1.0, 1.0]))
(array([-0.761594, 0.761594]), array([-0.158655, 0.841345]), array([-0.268941, 0.731059]))
```

## Constraints

Allowed: NumPy and Python standard library, including math.erf.
Not allowed: torch, jax, scipy, sklearn. Implement the formulas from scratch.

## Done looks like

The examples and edge cases pass. Hidden tests cover a seeded independent oracle, degenerate
inputs, output types, invalid arguments, scale extremes within the contract and input preservation.
The local derivatives of these functions are implemented and checked by independent finite
differences in the optional p02 problem.
