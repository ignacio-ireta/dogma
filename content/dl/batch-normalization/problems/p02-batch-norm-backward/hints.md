## Hint 1 — concept

The output of a normalized feature depends on every row through the batch mean and variance. Gradients therefore flow back into all rows, not only into the row whose output they came from. The derivation is one step of the forward pass, reversed carefully.

## Hint 2 — formula / approach

With $\hat x=(x-\mu)/\sigma$ and $\sigma=\sqrt{\mathrm{var}+\epsilon}$ (biased variance), the parameter gradients are $d\beta=\sum_i g_i$ and $d\gamma=\sum_i g_i\,\hat x_i$, where $g=\mathrm{grad\_y}$. The input gradient is $dx=\frac{\gamma}{N\sigma}\left(N g-\sum_i g_i-\hat x\,\sum_i g_i\hat x_i\right)$, where the sums run over rows.

## Hint 3 — code shape

Validate first, then compute $\hat x$ and $\sigma$ once. Compute the two per-feature sums over rows, `g.sum(dim=0)` and `(g * x_hat).sum(dim=0)`. Build `grad_x` from those sums and the scale $\gamma/(N\sigma)$. Check a one-row-per-feature case against a scalar loop before you trust the broadcast.
