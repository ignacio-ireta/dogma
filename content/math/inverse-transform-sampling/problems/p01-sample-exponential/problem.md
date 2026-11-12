# Exponential samples by inverse transform

Implement `sample_exponential(rate, n, rng)`. It returns `n` draws from the exponential distribution
with rate `rate` (mean `1 / rate`), by the inverse-transform method: each uniform draw `u` is mapped
through the inverse CDF, `F^-1(u) = -ln(1 - u) / rate`. The map is fixed, so the output is a
function of the generator's stream and can be reproduced exactly.

## Contract

- `rate`: a real number that is finite and greater than 0.
- `n`: an int, `n >= 0`.
- `rng`: a NumPy `Generator`. The function makes exactly one call, `u = rng.random(n)`, and no other
  draw. (In tests `rng` may be any object with a `random(n)` method that returns `n` values in
  `[0, 1)`.)
- Returns: a float64 array of shape `(n,)` whose entries are `-log1p(-u) / rate`, elementwise.
- Raises `ValueError` when `rate` is not finite and positive, or when `n` is negative.
- Must not mutate its inputs.

## Examples

```python
>>> sample_exponential(2.0, 3, np.random.default_rng(0))
array([0.5066, 0.1572, 0.0209])     # u = [0.6370, 0.2698, 0.0410] from rng.random(3)
>>> sample_exponential(1.0, 0, np.random.default_rng(0))
array([], dtype=float64)
>>> sample_exponential(0.0, 5, np.random.default_rng(0))
ValueError                          # the rate must be positive
```

## Constraints

- Allowed: NumPy (`np.log1p`, `np.asarray`, `math.isfinite`, …) and the `rng` you are given.
- Not allowed: `scipy`, and NumPy's or the standard library's exponential samplers, such as
  `rng.exponential`. The point is the inverse map applied to uniforms.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: exact agreement with the
formula applied to `rng.random(n)` for random seeds and rates; the formula for uniforms supplied
directly, including values very close to 0 and 1; one call to `rng.random(n)`; the scaling law
(doubling the rate halves every draw under the same seed); a Kolmogorov–Smirnov goodness-of-fit
check at a fixed seed; the shape and dtype for empty and non-empty output; and validation of `rate`
and `n`.

Some hidden tests take their inputs from the real dataset `online_retail` (see `docs/DATASETS.md`).
