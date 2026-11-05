## Hint 1 — concept

Quantization is choosing a grid of integers and a scale that maps each float onto the grid. With
symmetric absmax, the scale is set by the largest magnitude in a group, so that value lands on the
edge of the grid. A channel axis gives each slice its own grid.

## Hint 2 — formula / approach

For each group, `scale = absmax / 127`, and a group whose absmax is zero uses scale 1.0. Codes are
`round_half_to_even(x / scale)`, clipped to [-127, 127]. Dequantization multiplies back. Per-channel
means the absmax runs over every axis except the channel axis, so the scale keeps that axis.

## Hint 3 — code shape

Validate the dtype, the ndim, the emptiness and the finiteness first. Compute the absmax with
`np.max(np.abs(...), axis=..., keepdims=True)`, where `axis` is a tuple of every axis except the
channel axis (or None). Replace zero absmax with 1.0 using `np.where`. Divide, round, clip, then
`astype(np.int8)`. Reshape the scale to `()` or `(C,)` before returning.
