## Hint 1 — concept

AMSGrad keeps the Adam averages and adds one memory: the largest second moment seen so far. The
denominator uses that maximum, so the step cannot grow back after a large gradient just because
recent gradients are small.

## Hint 2 — formula / approach

Track `v_max` per coordinate with an elementwise maximum against the new `v`. Bias-correct the
first moment as in Adam. Bias-correct `v_max` as well, then divide by the root plus `eps`. The
maximum starts at zero and is carried in the state.

## Hint 3 — code shape

In the per-tensor loop, compute `m` and `v`, then `vmax = torch.maximum(vmax_prev, v)`. Build the
step from `m_hat` and `sqrt(vmax_hat) + eps`. Return the new parameters and the four-part state
`(new_m, new_v, new_vmax, t_new)`.
