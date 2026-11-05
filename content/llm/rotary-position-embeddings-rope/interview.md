# Interview notes — Rotary position embeddings (RoPE)

## Q1. How does RoPE encode position in an LLM attention layer?

**60-second answer.** Attention needs position, and RoPE puts it into the score rather than into the
embedding. Inside each head, the query and key are rotated by an angle that depends on the token's
absolute position. Pair $i$ of the feature axis rotates by $p\,\theta_i$, with $\theta_i = 10000^{-2i/d}$.
After rotation, the dot product of a query and a key depends only on the difference of their positions.
Values are not rotated, because they are what gets averaged, and the score is where relative position
belongs. In LLaMA-style code the pairs are half-split, feature $i$ with $i + d/2$.

**Follow-ups.** Why only q and k? The score is the only place where relative position is needed. What does
the layout change? The pairs. Half-split and interleaved are different functions, so a checkpoint needs its
own layout, and the two differ by one fixed permutation.

## Q2. How do positions work during cached decoding?

**60-second answer.** The cache holds keys that were already rotated at their absolute positions. A new
token's position is `past_len`, the number of cached tokens, so its query and key are rotated at that index.
If the new token were rotated at 0, its score against the cache would be wrong by a mismatch in angle, and
the output would look plausible and still be wrong. A useful test is that a one-token step with the offset
reproduces the last row of a full pass.

**Follow-ups.** Do cached keys need rerotating later? No, because their positions do not change. What if
the sequence is longer than the trained length? That is the context-extension question in Q3.

## Q3. How do you extend the context of a RoPE model?

**60-second answer.** Linear position interpolation divides every position by a scale $s \ge 1$. The model
sees positions it was trained on, and the window it can read grows by about $s$. The cost is that neighbouring
tokens sit closer together in angle than in training, so the model usually needs some fine-tuning. Changing
the base is another route, with its own trade-off. I would state both, and say which I measured.

**Follow-ups.** Does interpolation change the highest-frequency pair? It rotates every pair by the same scaled
position, so the fastest pair is compressed too. Why not scale only the low-frequency pairs? That is a
different method, and it is worth a separate topic.

## Common mistakes

- Rotating the value tensor as well as q and k.
- Using 0 as the first position of a decoding step instead of `past_len`.
- Loading a half-split checkpoint into an interleaved implementation without permuting, which runs without error.
- Dividing the frequency by the scale instead of the position, which changes the base silently.
