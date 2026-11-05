## Hint 1 — concept

The derivative is the slope of the best local linear approximation. A finite difference measures
the slope of a chord over a small interval of width `h`. Two errors fight each other. The chord is
not the tangent, so truncation error shrinks as `h` shrinks. Subtracting two nearly equal function
values loses digits, so round-off error grows as `h` shrinks. Symmetric (central) differences
cancel the first-order term, which is why they are far more accurate than one-sided ones.

## Hint 2 — formula / approach

Forward: `(f(x + h) - f(x)) / h`. Central: `(f(x + h) - f(x - h)) / (2 * h)`. The default step is
`eps**0.5 * max(1, abs(x))` for forward and `eps**(1/3) * max(1, abs(x))` for central, with
`eps = np.finfo(np.float64).eps`. Validate in this order: the method, then `x`, then `h`.

## Hint 3 — code shape

Check `method` against the two allowed strings and raise `ValueError` otherwise. Convert `x` with
`float(x)` and check `math.isfinite`. If `h is None`, compute the default step from the method;
otherwise check that `h` is positive and finite. Finish with one branch per method, and wrap the
result in `float(...)`.
