# Symmetric absmax int8 quantization

Implement two functions. `absmax_quantize` maps a real array to int8 codes with a symmetric
scale, either for the whole tensor or for each channel along one axis. `absmax_dequantize`
maps the codes back to floats. The scale is `absmax / 127`, so the largest magnitude in each
group lands on code 127 or -127, and the code -128 is never produced.

## Contract

`absmax_quantize(x, axis=None)`

- `x`: real NumPy array (integer or float dtype), `ndim >= 1`, at least one element, every value
  finite. Bool, complex and object arrays are invalid. Do not mutate `x`.
- `axis`: `None` for per-tensor scaling, or an `int` in `[-ndim, ndim)` naming the **channel
  axis**. With an axis, each index along it has its own scale, and the absmax is taken over
  all the other axes. Negative axes count from the end. `bool` is not an axis.
- Per tensor: `absmax = max(|x|)` over everything, `scale = absmax / 127`.
- Per channel: `absmax_c = max(|x[..., c, ...]|)` over the other axes, `scale_c = absmax_c / 127`.
- Zero group: if a group's absmax is 0 (the whole tensor, or one channel), its scale is `1.0`.
  Its codes are all 0.
- Underflow: a non-zero group whose `absmax / 127` is not a normal float64 (below
  `2.2250738585072014e-308`, the smallest normal) raises `ValueError`. Such a scale has too
  few significant bits to divide by, and it can underflow to 0.
- Codes: `q = clip(round_half_to_even(x / scale), -127, 127)`, computed in float64.
- Returns `(q, scale)`:
  - `q`: `numpy.int8` array with the same shape as `x`.
  - `scale`: float64 array. Shape `()` when `axis` is `None`, shape `(x.shape[axis],)` otherwise.
- Raises `ValueError` for an invalid dtype, `ndim == 0`, an empty array, a nonfinite value,
  an invalid axis, or an underflowing scale.

`absmax_dequantize(q, scale, axis=None)`

- `q`: `numpy.int8` array. Any other dtype raises `ValueError`.
- `scale`: float64 array, finite and strictly positive. Its shape must match `axis` exactly
  (`()` for `None`, `(q.shape[axis],)` otherwise). A mismatch raises `ValueError`.
- Returns `q * scale` as a float64 array, with the per-channel scale broadcast along `axis`.
  Shape equals `q.shape`.

## Examples

```python
>>> q, s = absmax_quantize(np.array([127.0, 0.5, 1.5, 2.5, -2.5]))
>>> q, s
(array([127, 0, 2, 2, -2], dtype=int8), array(1.))   # scale = 127 / 127; ties go to the even code

>>> q, s = absmax_quantize(np.array([[127.0, 1.0, -2.0], [0.5, -0.3, 0.1]]), axis=0)
>>> q
array([[127,   1,  -2],
       [127, -76,  25]], dtype=int8)
>>> s                                     # row 1 has its own scale, 0.5 / 127 = 0.0039370...
array([1.        , 0.00393701])
# A single per-tensor scale (127 / 127 = 1.0) would give row 1 the codes [0, 0, 0].

>>> absmax_quantize(np.zeros(3))
(array([0, 0, 0], dtype=int8), array(1.))  # all-zero group: scale 1.0, codes 0
```

## Constraints

- Allowed: the Python standard library and NumPy (including `np.round`, `np.rint`, `np.clip`,
  `astype` and `np.where`).
- Not allowed: `torch`. Its quantization functions do the whole job, so the lesson is lost.
  Write the scale, rounding and clipping yourself.
- Round half to even, the NumPy default. Do not use `floor(x + 0.5)`, which rounds halves up.

## Done looks like

All public examples pass. The hidden tests cover: an independent pure-Python model of per-tensor
and per-channel quantization on seeded random inputs (including negative axes); exact ties
resolved to even codes; the error bound `|x - x_hat| <= scale / 2`; codes that stay inside
`[-127, 127]` with `int8` dtype, and absmax entries that reach `+-127`; all-zero tensors and
all-zero channels next to non-zero ones; integer input; very small and very large magnitudes;
invalid dtypes, shapes, axes, empty and nonfinite inputs; inputs that are not mutated; and the
shape rules of the dequantizer.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).
