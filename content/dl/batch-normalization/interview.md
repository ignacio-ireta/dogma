# Interview notes — Batch normalization

## Q1. What does batch normalization do, and why does it behave differently in training and inference?

**60-second answer.** For each feature, training computes the mean and variance over the mini-batch, normalizes with them, then applies a learned scale and shift. Inference cannot use the batch, so it uses running averages collected during training. That split is the main design decision: the output for one example in training depends on its batch, but at inference it should not. I always state which mode I mean before I describe the formula.

**Follow-ups.** Why keep learned gamma and beta? They let the layer represent the identity, so normalization does not reduce capacity. What if I forget to switch to eval? Predictions depend on batch composition, so the same input gives different outputs across batches.

## Q2. Why does the running variance use N - 1 while the normalizing variance uses N?

**60-second answer.** The normalizing step should standardize the batch I actually have, so it uses the biased variance, dividing by N. The running estimate is meant to estimate the population variance. Averaging biased batch variances underestimates it by a factor of (N - 1) / N, so the running update uses the unbiased value. The difference matters for small batches, and it is a common source of silent mismatch between two implementations.

**Follow-ups.** Which convention does PyTorch use for momentum? The new batch statistic is weighted by momentum, with default 0.1. Some papers and frameworks weight the old value instead, so check the definition before you copy a number.

## Q3. When would you not use batch normalization?

**60-second answer.** Batch statistics are unreliable with small batches, batch size one is undefined in training, and the method depends on examples being independent within a batch. Sequence models with varying lengths, online inference, and some distributed setups are poor fits. Layer normalization removes the batch dependence, so it is the usual choice for transformers.

**Follow-ups.** What about a batch of 32 images per GPU across 8 GPUs? The statistics are per device unless you synchronize them. That is a reasonable question to raise, but I would verify the actual setup before claiming a problem.

## Common mistakes

- Normalizing over the feature axis instead of the batch axis for an (N, C) input.
- Using biased variance in the running update, or unbiased variance in the normalization.
- Forgetting eval mode, so inference uses the batch's own statistics.
- Treating the momentum argument as the weight on the old running value, which is the opposite of the PyTorch convention.
