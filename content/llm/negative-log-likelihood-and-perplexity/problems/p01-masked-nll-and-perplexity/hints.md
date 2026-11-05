## Hint 1 — concept

Each scored token contributes the negative log of its softmax probability. Averaging is over the scored tokens only, because ignored positions are not predictions of the text. Perplexity is the exponential of that average, so it is computed last. Think about which of these steps can overflow, and which need a count that excludes some positions.

## Hint 2 — formula / approach

For a position with logits z and target y, the NLL is LSE(z) - z[y], where LSE(z) = log of the sum of exp(z). To avoid overflow, subtract the maximum m of z: LSE(z) = m + log of the sum of exp(z - m). The mean is the sum of NLL over scored positions divided by the number of scored positions. Perplexity is exp of that mean.

## Hint 3 — code shape

Build a boolean mask `targets != ignore_index`. Compute the row maximum over the last axis with keepdim, subtract it, exponentiate, sum over the last axis, take the log and add the maximum back. Gather the target logit with an index tensor whose ignored entries are clamped to a valid index. Multiply the per-position NLL by the mask, sum, and divide by the mask count. Validate the targets range before any gather.
