## Hint 1 — concept

Momentum keeps a running direction. Each step adds the new gradient to a decayed copy of the
old direction, and the parameters move along that running direction, not the raw gradient.

## Hint 2 — formula / approach

Keep one buffer `v` per tensor with `v = momentum * v_prev + g`. Learning rate enters only in
`p - lr * v`. On the first step there is no previous buffer, so the first buffer is the raw
gradient. Check the PyTorch buffer convention against your returned state, not just the new
parameters.

## Hint 3 — code shape

Validate lengths and shapes first. Then loop over `zip(params, grads)`. Inside the loop, choose
the buffer with an `if velocity is None` branch, append it to a new list, and append
`p - lr * buf` to a second list. Return both lists. Use out-of-place operations so the inputs
stay unchanged.
