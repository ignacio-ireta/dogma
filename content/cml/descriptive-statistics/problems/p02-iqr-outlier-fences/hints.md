## Hint 1 — concept

A quartile is a position in the sorted data. When the position falls between two values, the quartile
is the interpolation between them. Fences built from quartiles depend only on the values near the
quartiles, so an extreme value far away does not move them.

## Hint 2 — formula / approach

Sort a copy of the data. For q, compute `h = q * (n - 1)`, let `lo = floor(h)`, and interpolate
between `s[lo]` and `s[min(lo + 1, n - 1)]` using `h - lo`. Then `IQR = Q3 - Q1`, and the fences
are `Q1 - k * IQR` and `Q3 + k * IQR`.

## Hint 3 — code shape

`values = np.array(xs, dtype=np.float64)`, then the checks (1-D, non-empty, finite, `k >= 0`). Write a
helper `_quantile_linear(sorted_values, q)` that uses `math.floor` and returns a Python float. Call it
with `q = 0.25` and `q = 0.75` on `np.sort(values)`, and return `(q1 - k * iqr, q3 + k * iqr)`.
