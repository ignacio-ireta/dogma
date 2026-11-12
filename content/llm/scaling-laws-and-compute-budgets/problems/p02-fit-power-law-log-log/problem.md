# Fit a power law by log-log least squares

Scaling-law papers plot loss, or the optimal model size, against compute on log-log axes and read
off a straight line. A power law `y = a * x**b` becomes a line once both sides are logged:
`log y = log a + b * log x`. This function fits that line by ordinary least squares and returns
the two constants of the power law.

## Contract

- `fit_power_law(x, y)` with `x` and `y` as NumPy arrays:
  - Both are one-dimensional with the same length `n >= 2`. Any other rank raises `ValueError`, and
    so does a length mismatch.
  - Both hold real numbers. Integer and float dtypes are accepted; the function converts them to
    `float64` and does not change the caller's arrays.
  - Every `x_i > 0`, every `y_i > 0`, and every value is finite. Otherwise `ValueError`, because
    the logarithm is undefined.
  - Not all `x_i` are equal, otherwise the slope is undefined and the function raises `ValueError`.
- Method. Let `u_i = log(x_i)` and `v_i = log(y_i)` (natural log). Then
  - slope `b = sum((u - mean u) * (v - mean v)) / sum((u - mean u)**2)`,
  - intercept `c = mean v - b * mean u`,
  - `a = exp(c)`.
  Use the centred sums. The expanded form `sum(u*v) - n*mean(u)*mean(v)` cancels badly when the
  `u_i` are close together and large in magnitude.
- Returns a tuple `(a, b)` of two Python `float`s, with `a > 0`.
- The fit minimises the squared residuals in log space only. It is not a fit in linear space.

## Examples

```python
>>> x = np.array([1.0, 2.0, 4.0, 8.0])
>>> y = 2.0 * np.sqrt(x)                  # y = 2 * x**0.5
>>> a, b = fit_power_law(x, y)
>>> round(a, 9), round(b, 9)
(2.0, 0.5)
```

```python
>>> a, b = fit_power_law(np.array([1.0, 10.0, 100.0]), np.array([5.0, 5.0, 5.0]))
>>> round(a, 9), round(b, 9)
(5.0, 0.0)                               # a flat line in log space: b = 0 (a is 4.999... unrounded)
```

```python
>>> fit_power_law(np.array([3.0, 3.0, 3.0]), np.array([1.0, 2.0, 4.0]))
ValueError                               # all x equal: the slope is undefined
```

## Constraints

- Allowed: NumPy array operations on the input (`numpy.log`, `numpy.exp`, `numpy.asarray`, sums and
  means), and the `math` module.
- Not allowed: `numpy.polyfit`, `numpy.linalg` in any form (so no `lstsq`, `solve` or `inv`), and
  `scipy`, `sklearn` and `statsmodels`. Those are the library fits this problem asks you to write.

## Done looks like

All public tests pass. Submit then runs the hidden tests. They cover agreement with an independent
least-squares fit (`numpy.polyfit` on the logs) on seeded noisy data; exact recovery of noise-free
laws with negative and positive exponents; the scaling rules (multiplying `y` scales `a`, multiplying
`x` rescales `a` by a power of the factor); inputs with huge and tiny magnitudes; inputs that sit far
from zero in log space, where the expanded formula fails; two points; the error rules (length, rank,
single point, equal `x`, non-positive or non-finite values); integer arrays; the return type; and
inputs left unchanged.

The hidden tests draw their bulk inputs from the real datasets `gutenberg_frankenstein`,
`gutenberg_pride_prejudice` and `gutenberg_walden` (see `docs/DATASETS.md`).
