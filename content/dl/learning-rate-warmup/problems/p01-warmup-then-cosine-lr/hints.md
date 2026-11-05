## Hint 1 — concept

The schedule has two pieces with one boundary. Early on, the rate rises in a straight line from zero.
After the ramp, the rate falls on a half-cosine. Decide which piece a given step belongs to before
you compute anything.

## Hint 2 — formula / approach

Let `W` be the warmup length and `T` the total length. For `step < W`, the rate is the base rate times
`step / W`. For `W <= step < T`, measure progress `p` from zero at `W` to one at `T`, then blend from
the base rate down to the floor with the factor `(1 + cos(pi * p)) / 2`. At `step >= T`, return the floor.

## Hint 3 — code shape

Validate the arguments first, with the boundary rules for `total_steps > warmup_steps`. Then use an
if-branch for the ramp, an if-branch for the floor, and a final cosine branch. Test the value at
`step == warmup_steps` by hand, since that is where a branch can be off by one.
