# Matmul roofline: bound and time floor

Implement `matmul_roofline(m, n, k, element_bytes, peak_flops, bandwidth)`. It describes one dense product
`C = A B`, where `A` is `m x k` and `B` is `k x n`, on a device with peak compute `peak_flops` (FLOP/s) and memory
bandwidth `bandwidth` (bytes/s). It returns the work and traffic of the product, its roofline verdict and a lower
bound on its run time.

## Contract

Counting rules (use them exactly):

- FLOPs: `flops = 2 * m * n * k`. Each of the `m * n` outputs is a dot product of length `k`: `k` multiplies and `k` adds.
- Bytes: `bytes = (m * k + k * n + m * n) * element_bytes`. Each element of `A` and `B` is read once, each element of `C` is written once, and no cache reuse is credited.
- Intensity: `intensity = flops / bytes`, in FLOP per byte.
- Ridge: `peak_flops / bandwidth`, in FLOP per byte.
- Bound: `"compute"` when `intensity >= ridge`, otherwise `"memory"`. Decide it with exact integer arithmetic: `"compute"` when `flops * bandwidth >= peak_flops * bytes`. At exactly the ridge the bound is `"compute"`.
- Attainable FLOP/s: `peak_flops` when compute-bound, otherwise `intensity * bandwidth`.
- Time floor, in seconds: `max(flops / peak_flops, bytes / bandwidth)`.

Arguments:

- `m`, `n`, `k`: Python `int`, at least 1.
- `element_bytes`: Python `int`, at least 1. Bytes per element (fp16 is 2, fp32 is 4, fp8 is 1).
- `peak_flops`, `bandwidth`: Python `int`, at least 1. Floats are rejected, so write the device spec as integers.

Returns a `dict` with exactly these keys:

- `"flops"`: `int`
- `"bytes"`: `int`
- `"intensity"`: `float`
- `"attainable_flops"`: `float`
- `"bound"`: `str`, either `"memory"` or `"compute"`
- `"time_s"`: `float`, in seconds

No rounding: floats are returned at full precision.

Raises:

- `TypeError` when any argument is not a Python `int`. `bool`, `float` and `str` are rejected.
- `ValueError` when any argument is below 1.

Must not mutate inputs (there are none to mutate; do not keep module-level state).

## Examples

```python
>>> PEAK, BW = 312_000_000_000_000, 2_000_000_000_000        # ridge 156 FLOP/byte
>>> r = matmul_roofline(1, 4096, 4096, 2, PEAK, BW)          # batch-1 decode GEMV, fp16
>>> r["flops"], r["bytes"]
(33554432, 33570816)
>>> r["bound"], round(r["intensity"], 4)
('memory', 0.9995)
>>> r["time_s"]                                              # bytes / bandwidth, about 1.68e-5 s
```

```python
>>> matmul_roofline(4096, 4096, 4096, 2, PEAK, BW)["bound"]  # intensity about 1365 > ridge 156
'compute'
```

```python
>>> matmul_roofline(150, 150, 150, 1, 1000, 10)             # intensity 100 == ridge 100: the tie
{'flops': 6750000, 'bytes': 67500, 'intensity': 100.0, 'attainable_flops': 1000.0, 'bound': 'compute', 'time_s': 6750.0}
```

```python
>>> matmul_roofline(0, 4, 4, 2, PEAK, BW)                   # m must be at least 1
ValueError
```

## Constraints

- Allowed: built-in types and the standard library, for example `math` or `fractions`.
- Not allowed: `numpy`, and any helper that computes matmul costs or the roofline for you. Keep the counts as
  Python integers so they stay exact.

**Why integers.** A float holds only about 16 significant digits. `float(2**70 + 1)` is `2**70`, so a product whose
counts pass 2**53 loses its low digits when they are multiplied as floats. The bound test `flops * bandwidth >= peak_flops * bytes`
is therefore done on Python ints, and only the final divisions for `intensity` and `time_s` produce floats.

## Done looks like

The public tests check the four examples above. **Submit** adds hidden tests. They compare your result with an exact
rational computation over seeded random shapes, element widths and devices, to a relative tolerance of 1e-9. They
also cover: the exact tie at the ridge; a ridge a hair above the intensity, which must stay memory-bound; a single-row
product; the smallest 1 x 1 x 1 product; large dimensions whose counts exceed float precision; the factor 2 in the
FLOP count; the output bytes in the traffic count; the time floor taken as the larger of the two times; the attainable
rate in both regimes; the exact key set and value types; and each validation rule.
