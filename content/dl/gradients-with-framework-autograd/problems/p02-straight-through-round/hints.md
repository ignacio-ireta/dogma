## Hint 1 — concept

The forward pass and the backward pass can use different functions. Round in the forward pass. In the backward pass, pretend the rounding was not there and let the gradient through unchanged.

## Hint 2 — formula / approach

Let $Q(x) = \mathrm{round}(x/s)\,s$ be the staircase. You want an output whose value is $Q(x)$ and whose derivative with respect to $x$ is $1$. A term $x + (Q(x) - x)$ has value $Q(x)$. Its derivative is $1 + (Q' - 1)$, and $Q'$ is zero almost everywhere. Detaching the difference removes its contribution to the backward pass.

## Hint 3 — code shape

Compute the quantized tensor once with `torch.round(x / step) * step`. Return `x` plus the detached difference between that tensor and `x`. Validate `step` and the dtype before any tensor work.
