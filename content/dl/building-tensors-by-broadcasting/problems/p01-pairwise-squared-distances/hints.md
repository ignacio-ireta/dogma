## Hint 1 — concept

The distance between row `i` of `X` and row `j` of `Y` is one number, and you want all `n * m` of
them at once. Broadcasting makes that possible: give `X` a size-1 axis where `Y`'s points go, and
`Y` a size-1 axis where `X`'s points go. Then one subtraction covers every pair.

## Hint 2 — formula / approach

`D[i, j] = sum over k of (X[i, k] - Y[j, k])**2`. Build the difference for every pair first, then
sum over the feature axis. The shapes go (n, 1, d) minus (1, m, d) to (n, m, d), and the sum over
the last axis gives (n, m). Check that both inputs are 2-D and that their feature counts match
before you broadcast, because a size-1 feature axis would stretch without an error.

## Hint 3 — code shape

Convert both inputs to float arrays and check `.ndim` and `.shape[1]`. Then take a difference with
one new axis inserted in each operand (`x[:, None, :]` and `y[None, :, :]`), square it, and sum over
the last axis. Return that array directly.
