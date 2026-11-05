## Hint 1 — concept

Adam keeps two running averages per parameter: one of gradients, one of squared gradients. Both
start at zero, which biases them toward zero for the first few updates. The update divides the
first moment by the root of the second, so each coordinate moves by a bounded amount.

## Hint 2 — formula / approach

Update `m` and `v` with the exponential averages. Before dividing, correct each average by
`1 - beta**t`, where `t` is the number of updates including this one. Put `eps` after the square root
and after bias correction. Keep `t` in the state and increment it before you use it.

## Hint 3 — code shape

Start from `state is None` by creating zero lists and `t = 0`. Compute `t_new = t + 1` first.
Inside a loop over the tensors, update `m` and `v`, compute `m_hat` and `v_hat`, then build the new
parameter. Return the new parameters and the tuple `(new_m, new_v, t_new)`. Validate the arguments before
the loop.
