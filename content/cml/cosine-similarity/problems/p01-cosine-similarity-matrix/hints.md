## Hint 1: concept

Cosine is the dot product after each vector has been divided by its length. To compare every
pair at once, normalise each matrix row by row, then multiply the two normalised matrices. A zero
row needs its own rule, because dividing by its length is undefined.

## Hint 2: formula / approach

`S = Â @ B̂.T`, where each row of `Â` is `A[i] / ‖A[i]‖`. Find the rows whose norm is zero and set
their similarities to 0.0 after the product. Clip to `[-1, 1]`. For very large or very small
entries, divide each row by its largest absolute entry before computing the norm.

## Hint 3: code shape

Validate first: `ndim`, the column counts, empty input, and `np.isfinite`. Write a helper that
returns `(unit_rows, zero_mask)`. Then `sims = np.clip(unit_a @ unit_b.T, -1.0, 1.0)`, followed by
`sims[zero_a, :] = 0.0` and `sims[:, zero_b] = 0.0`.
