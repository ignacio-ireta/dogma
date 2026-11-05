## Hint 1 — concept

An asymmetric grid is shifted: the codes cover the real range [lo, hi] with 256 evenly spaced steps,
and a zero point records which code stands for 0.0. Make the range include zero first, so an
all-positive tensor still represents 0.0 exactly.

## Hint 2 — formula / approach

`lo = min(min(x), 0)`, `hi = max(max(x), 0)`, `scale = (hi - lo) / 255`, with `scale = 1.0` when the
two bounds are equal. The zero point is the integer that sends `lo` to -128:
`zero_point = -128 - round(lo / scale)`. A code is `round(x / scale) + zero_point`, clipped to
[-128, 127]. Dequantize with `scale * (q - zero_point)`.

## Hint 3 — code shape

Compute `lo` and `hi` from `min` and `max` of the flattened array, each combined with 0.0. Derive
`scale` and an integer `zero_point`, then compute codes with one vectorized line. Cast the codes to
`int8` at the end, and return `float(scale)` and `int(zero_point)` as plain Python numbers.
