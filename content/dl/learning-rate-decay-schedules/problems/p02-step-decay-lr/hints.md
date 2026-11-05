## Hint 1 — concept

The rate is constant inside a block of updates and then drops by one fixed factor. Count how many
complete blocks have passed. The count only changes at multiples of the block length.

## Hint 2 — formula / approach

Let `s` be the block length and `t` the step. The number of drops is the number of completed
blocks, `floor(t / s)`, which is integer division. Each drop multiplies by `gamma`, so the rate is
`base_lr` times `gamma` raised to that count.

## Hint 3 — code shape

Validate the four arguments first. Then compute the block count with integer division, and
return `base_lr * gamma ** count` as a float. Do not loop over the steps.
