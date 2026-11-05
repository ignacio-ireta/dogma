## Hint 1 — concept

Each feature is normalized on its own, using statistics computed down the batch axis. Training and inference differ in where the statistics come from, and the running estimates are smoothed over many batches.

## Hint 2 — formula / approach

Training: $\mu_B$ is the mean over rows, $\sigma_B^2$ is the variance over rows with divisor $N$, and $y=\gamma\,\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}+\beta$. The running variance is updated with the same batch variance rescaled by $\frac{N}{N-1}$. In both running updates, the new batch statistic is weighted by momentum. Inference uses the stored statistics in the same $y$ formula.

## Hint 3 — code shape

Validate first: rank, dtype, N, the `(C,)` shapes, momentum and eps. Then branch on `training`. Compute `mean(dim=0)` and the centred squares. Build the outputs as new tensors, so nothing you return aliases an input. Check that a one-row training call raises before any arithmetic.
