# Interview notes — Positional encodings

## Q1. Why does a Transformer need positional encodings?

**60-second answer.** Attention computes a weighted sum over all positions, and the weights depend only on the content of the query and key vectors. If you permute the input tokens, the output permutes the same way. So without position information the model is a bag of tokens. The fix is to put position into the representation: add a position vector to the embedding, or rotate queries and keys by position. I would name the trade-off first: absolute schemes are simple, while rotary schemes make the score depend on the relative offset directly.

**Follow-ups.**

- Why sines and cosines? They give a fixed, deterministic code with a unit-norm pair per frequency, and $\mathrm{PE}_{pos+k}$ is a linear function of $\mathrm{PE}_{pos}$ for each pair.
- Learned or sinusoidal? Vaswani et al. report nearly identical quality for learned embeddings in their experiment, and they chose the sinusoidal form because it may extrapolate to lengths not seen in training. That was a hypothesis in the paper, not a tested result.

## Q2. How does RoPE make the attention score depend on relative position?

**60-second answer.** RoPE rotates each adjacent pair of query and key features by an angle proportional to the token position. The rotations for positions $m$ and $n$ combine into a rotation by $(m-n)\theta$, so the dot product depends only on the offset. Norms are preserved, so the rotation does not change the scale of the vectors. Values are left unrotated.

**Follow-ups.**

- Where is it applied? To queries and keys only, after the linear projections and before the dot product.
- What limits it? The model only learns angles it has seen in training, so extending the context usually means rescaling the frequencies.

## Common mistakes

- Adding the table in the wrong order: sines in the first half of the columns and cosines in the second half. That is a different function, and a model trained with it cannot reuse a standard table.
- Using the exponent $i$ instead of $2i$ with the $d_{model}$ denominator, which changes every frequency.
- Rotating split halves $(x_{0..d/2}, x_{d/2..d})$ when the convention says adjacent pairs, or the other way round, and then comparing against a library that uses the other layout.
- Applying RoPE to values as well; it belongs on queries and keys.
