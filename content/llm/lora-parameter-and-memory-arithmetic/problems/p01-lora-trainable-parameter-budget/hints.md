## Hint 1 — concept

Count one adapted matrix at a time. Each matrix has an input width and an output width, and its
LoRA pair holds a number for every row and column of the bottleneck. The attention projections are
square, the two MLP input projections go from the model width to the MLP width, and the MLP output
projection goes back. Then multiply by the number of layers, because every layer repeats the same set.

## Hint 2 — formula / approach

For a matrix with input width `d_in` and output width `d_out`, the adapter holds
`rank * (d_in + d_out)` numbers. Pick `(d_in, d_out)` from the target name: `d_model, d_model` for
`q k v o`, `d_model, d_ff` for `gate up`, and `d_ff, d_model` for `down`. The trainable count is
`n_layers` times the sum over targets. The fraction is that count divided by `base_params`.

## Hint 3 — code shape

Validate the arguments first: each integer is a non-bool `int` in range, `targets` is a non-empty
list or tuple with no repeats and only known names, and `d_ff` is present when an MLP target is
given. Loop over `targets`, use a small `if` chain to set `d_in` and `d_out`, and add
`rank * (d_in + d_out)` to a per-layer total. Multiply by `n_layers` once at the end. Return the
integer and the float fraction as a tuple.
