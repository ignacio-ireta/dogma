# Double all values

Implement `double_all(values)`, which returns a new list with every value doubled.

## Contract

- `values`: a list of real numbers. May be empty.
- Returns: a new list of floats with the same length and order.
- Raises: nothing for valid input.
- Must not mutate its input.

## Examples

```python
>>> double_all([1, 2.5])
[2.0, 5.0]
>>> double_all([])
[]
```

## Constraints

- Allowed: the standard library.   Not allowed: none.

## Done looks like

All public tests pass, then the hidden tests pass. They check agreement with NumPy on random
input, empty input, large and small magnitudes, and that the input is left unchanged.
