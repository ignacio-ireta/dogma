## Hint 1 — concept

After sampling, dropout is a deterministic elementwise multiplication. The same map must serve forward and backward.

## Hint 2 — formula / approach

Keep a unit when its sampled uniform is at least p. Divide the keep indicator by 1-p. Evaluation uses a multiplier of one.

## Hint 3 — code shape

Validate mode and seed, create one local generator in training, and build one multiplier array. Apply it to X and G. Return copies in evaluation rather than input aliases.
