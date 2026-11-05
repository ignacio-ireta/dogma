# Interview notes — Layer normalization

## Q1. How is layer normalization different from batch normalization, and why does that matter for sequences?

**60-second answer.** Batch normalization reduces over the batch, per feature. Layer normalization reduces over the features of each example, per position. The consequence is that one example's output does not depend on other examples, so there are no running statistics and training and inference compute the same thing. For sequences, the batch mixes different lengths and padding, so batch statistics are not meaningful. Layer normalization avoids all of that.

**Follow-ups.** Does it need a minimum batch size? No, it works with batch size one. What does it normalize over in a transformer? The hidden dimension of each token, not the sequence.

## Q2. Where does normalization sit in a transformer block, and what changes with pre-norm?

**60-second answer.** The original placement puts normalization after the residual add (post-norm). Pre-norm places it before each sublayer, inside the residual branch. I would describe the trade-off at this level: placement changes how signal flows through depth, and I would check the details of the specific architecture before I rely on a claim about training stability. The residual mechanism itself is covered in the next topic.

**Follow-ups.** Why does the placement matter at all? It changes how the input signal reaches the output in a deep stack. Is the choice fixed? No, it is an architectural decision that comes with its own training recipe.

## Common mistakes

- Normalizing over the batch axis, which turns the layer into batch normalization.
- Forgetting that the gain and bias are per feature, so they must broadcast over the last dimension only.
- Using the unbiased variance, which changes the output for every row.
