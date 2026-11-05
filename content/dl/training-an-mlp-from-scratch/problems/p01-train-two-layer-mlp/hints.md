## Hint 1 — concept

A training step uses one fixed parameter snapshot for its entire backward pass. Keep predictions, gradients, and updates as separate stages.

## Hint 2 — formula / approach

The prediction gradient is (prediction-y)/n. Send it through the output weights, then multiply by 1-H**2. Each affine map yields a weight contraction and a batch-summed bias.

## Hint 3 — code shape

Initialize a local generator, draw weights in the specified order, and loop over checkpoints from zero through steps. Record loss before deciding whether another update is needed. Cache all gradients before assignments.
