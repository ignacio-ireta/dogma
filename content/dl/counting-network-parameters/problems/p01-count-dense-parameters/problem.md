# Count a dense network without allocating it

Implement a parameter-budget calculator for a chain of distinct dense layers. The input lists
the input width followed by each layer's output width. Count stored scalar parameters, independently
of the number of examples or activation values. No layers share their parameters.

## Contract

- `widths`: a list or tuple of at least two positive **Python integers**; booleans, float-valued
  integers and NumPy integer scalars are not accepted. Each neighboring pair defines one layer.
- `use_bias`: a Python `bool`, default `True`, applying to every layer.
- Return `(per_layer, total)`: a list of Python integer counts in forward order, followed by
  their exact Python integer sum. A layer from width `a` to width `b` has `a*b` weights plus
  `b` biases when enabled. Count the output layer as well as hidden layers.
- Raise `ValueError` for invalid widths, fewer than two widths or a non-bool flag.
- Do not mutate `widths`. Do not allocate weight matrices. Arbitrarily large positive Python
  integers are supported by integer arithmetic; counts must not wrap at fixed-width limits.

## Examples

```python
>>> count_dense_parameters([3, 4, 2])
([16, 10], 26)
```

```python
>>> count_dense_parameters((3, 4, 2), False)
([12, 8], 20)
```

## Constraints

Allowed: Python standard library.
Not allowed: numpy, torch, jax, sklearn. Implement the formulas from scratch.

## Done looks like

The examples and edge cases pass. Hidden tests cover a seeded independent oracle, degenerate
inputs, output types, invalid arguments, scale extremes within the contract and input preservation.
