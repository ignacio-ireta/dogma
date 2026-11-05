## Hint 1 — concept

Nesterov momentum looks ahead before it steps. Weight decay is a penalty gradient, so it
belongs in the gradient before the momentum buffer is formed.

## Hint 2 — formula / approach

Start from `d = g + weight_decay * p`. Update the buffer as in plain momentum, with `d` as the
gradient. Then take the step from a look-ahead: `d + momentum * v`, not from `v` alone. Ask
which quantity the returned velocity should hold.

## Hint 3 — code shape

Inside the per-tensor loop, compute `d` first. Use an `if velocity is None` branch for the
buffer, compute the look-ahead from the new buffer, and append `p - lr * look_ahead` to the
output list. Validate `momentum` against the open interval `(0, 1)`.
