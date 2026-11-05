## Hint 1 — concept

AdamW separates two things Adam with L2 mixes together. The adaptive step is computed from the
gradient alone. The weight decay is a direct shrink of the weights, applied outside the adaptive
scaling.

## Hint 2 — formula / approach

Multiply each weight by `1 - lr * weight_decay` before the Adam update, using the value before this
update. Keep the raw gradient untouched in the moment updates. The Adam part is exactly the Adam
update from the previous topic, with the same counter and bias correction.

## Hint 3 — code shape

Validate the arguments first. Compute `shrink = 1 - lr * weight_decay` once. Inside the per-tensor
loop, build `p * shrink`, update `m` and `v` from `g`, and subtract the adaptive step from the shrunk
value. Return the new parameters and `(new_m, new_v, t_new)`.
