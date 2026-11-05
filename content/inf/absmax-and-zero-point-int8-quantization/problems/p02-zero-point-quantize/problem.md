# Asymmetric zero-point int8 quantization

Implement `zero_point_quantize(x)`. It maps a real tensor to the full signed int8 range
`[-128, 127]` with one scale and one integer zero point. Unlike symmetric absmax, the grid is
shifted, so a tensor that is mostly positive (for example after a ReLU) uses all 256 codes. The
range always contains 0.0, so zero is exactly representable.

## Contract

`zero_point_quantize(x)`

- `x`: real NumPy array (integer or float dtype), `ndim >= 1`, at least one element, every value
  finite. Bool, complex and object arrays are invalid. Do not mutate `x`.
- Range: `lo = min(min(x), 0.0)` and `hi = max(max(x), 0.0)`. Both bounds include zero.
- Scale: `scale = (hi - lo) / 255`. If `lo == hi`, which happens only when every value is 0,
  `scale = 1.0`.
  If `(hi - lo) / 255` is below the smallest normal float64 (`2.2250738585072014e-308`), raise
  `ValueError`, as for absmax.
- Zero point: `zero_point = -128 - round_half_to_even(lo / scale)`. This integer is the code of
  the real value 0.0.
- Codes: `q = clip(round_half_to_even(x / scale) + zero_point, -128, 127)`.
- Dequantization is `x_hat = scale * (q - zero_point)`.
- Returns `(q, scale, zero_point)`:
  - `q`: `numpy.int8` array with the shape of `x`.
  - `scale`: Python `float`, strictly positive.
  - `zero_point`: Python `int`, never a `bool`, in `[-128, 127]`.
- Raises `ValueError` for an invalid dtype, `ndim == 0`, an empty array, a nonfinite value, or an
  underflowing scale.

## Examples

```python
>>> zero_point_quantize(np.array([-100.0, 0.0, 155.0]))
(array([-128,  -28,  127], dtype=int8), 1.0, -28)
# lo = -100, hi = 155, scale = 255 / 255 = 1.0. zero_point = -128 - round(-100) = -28.

>>> zero_point_quantize(np.array([1.0, 2.0, 255.0]))
(array([-127, -126,  127], dtype=int8), 1.0, -128)
# All values are positive, so lo = 0 (the range is forced to contain 0.0) and zero_point = -128.

>>> zero_point_quantize(np.zeros(2))
(array([-128, -128], dtype=int8), 1.0, -128)
# Degenerate range: scale 1.0. Dequantizing gives exactly 0.0.
```

## Constraints

- Allowed: the Python standard library and NumPy (including `np.round`, `np.clip`, `astype`).
- Not allowed: `torch`. Its quantized-tensor utilities compute the affine parameters for you.
- Round half to even, as in the absmax problem. Keep the zero point an integer.

## Done looks like

All public examples pass. The hidden tests compare codes, scale and zero point with an independent
pure-Python model on seeded random inputs, including shifted ranges. They check the error bound
`|x - x_hat| <= scale / 2` for every element, exact representation of 0.0 inside any range, the
degenerate all-zero tensor, a single positive value, and very small and very large magnitudes.
They also check that the zero point is an `int` in the int8 range, and that empty, nonfinite,
bool and complex inputs raise `ValueError`, and that inputs are not mutated.
