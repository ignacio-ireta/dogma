## Hint 1 — concept

The rate starts at the base value and falls smoothly to the floor over exactly `t_max` updates.
A cosine is the natural shape: it is flat at both ends and steepest in the middle. Ask which
value of the cosine's argument corresponds to the start and which to the end.

## Hint 2 — formula / approach

Let `t` be the step and `T` the half-period. Map `t / T` onto a cosine phase in `[0, pi]`, then
rescale `cos` from `[-1, 1]` to `[0, 1]`. That factor multiplies the distance between the base
rate and the floor, and the floor is added back. Handle `step > T` separately.

## Hint 3 — code shape

Validate every argument first and raise `ValueError` with a message that names the bad argument.
Then compute one cosine phase from the step and `t_max`, convert it to a weight between 0 and 1,
and blend the base rate with the floor using that weight. Check the endpoints by hand before you
trust the result.
