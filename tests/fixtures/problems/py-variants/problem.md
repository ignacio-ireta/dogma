# Add two numbers

Implement `add(a, b)`, which returns the sum of two numbers.

## Contract

- `a`, `b`: ints or floats. A bool is not a number here.
- Returns: `a + b`. Sums of ints are exact, whatever their size.
- Raises `TypeError` when an argument is not an int or a float.

## Examples

```python
>>> add(2, 3)
5
>>> add(0.5, 0.25)
0.75
```

## Constraints

- Allowed: the standard library.
- Not allowed: none.

## Done looks like

All public tests pass, then the hidden tests pass. They check other values, large ints and
arguments of the wrong type.
