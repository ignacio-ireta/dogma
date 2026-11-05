# Interview notes — Transformer block

## Q1. Walk through one pre-norm decoder block.

**60-second answer.** The block keeps a residual stream of shape $(B, T, D)$. The first sub-layer normalises its input, runs causal multi-head attention with $H$ heads of width $D/H$, and adds the result to the stream. The second sub-layer normalises the updated stream, applies a feed-forward network that expands to $F$ with GELU and projects back to $D$, and adds that to the stream. The mask sets future scores to negative infinity before the softmax, and the scores are divided by the square root of the head width.

**Follow-ups.** Where does dropout go? On each sub-layer output, before the residual addition. Why are the LayerNorms on the inputs, not the sums? So the residual path stays unscaled, which is the stability argument in Q2.

## Q2. Pre-norm or post-norm, and why?

**60-second answer.** Post-norm normalises $x + \mathrm{sublayer}(x)$, so the stream itself is normalised at every block. The original post-norm Transformer used a learning-rate warm-up. Pre-norm normalises only the sub-layer input and keeps an identity path through the stream. Xiong et al. (2020) analyse this placement and report that pre-norm trains stably without warm-up. The cost is an un-normalised output stream, so a final LayerNorm is common. I would choose pre-norm for a deep stack unless I have a specific reason for post-norm.

**Follow-ups.** Does pre-norm remove the need for warm-up entirely? Not guaranteed. Many recipes still use a learning-rate schedule, so I would check the training curve rather than assume. What does the final LayerNorm do? It normalises the stream before the output head, so the logits have a stable scale.

## Q3. How many parameters does one block have?

**60-second answer.** With biases and LayerNorm, one block has $4D^2 + 2DF + 9D + F$ parameters. The attention projections contribute $4D^2$, and the feed-forward weights contribute $2DF$. For $D=512$ and $F=2048$ that is about $3.15$ million per block. The feed-forward sub-layer holds about twice the attention projection weights when $F=4D$, since $2DF = 8D^2$ against $4D^2$.

**Follow-ups.** How does this change with the number of heads? It does not. The heads split the same $D$ features, so the projection sizes stay the same. What is the $F$ usually set to? Commonly about $4D$, though it is a design choice.

## Common mistakes

- Placing LayerNorm on the residual sum and calling it pre-norm.
- Applying the causal mask after the softmax, which leaves the weights unnormalised.
- Scaling scores by $\sqrt{D}$ instead of $\sqrt{d_\text{head}}$.
- Counting only the weight matrices and forgetting biases and LayerNorm parameters.
