# Interview notes — GPT forward pass

## Q1. Walk me through a decoder-only forward pass.

**60-second answer.** The input is a batch of token ids with shape $(B, T)$. Each id selects a row of
the token table, and the row for its position is added, which gives a $(B, T, D)$ stream. The stream
goes through $N$ pre-norm blocks. Each block adds a causal attention output and a feed-forward output,
and every sub-layer reads a normalised copy of the stream. A final LayerNorm normalises the stream
once more, and a head matrix of shape $(D, V)$ turns each position into $V$ scores. The result is
$(B, T, V)$ logits, and training applies a cross-entropy loss to them against the next tokens.

**Follow-ups.** Where does the causal mask act? Inside each attention sub-layer, on the scores, before
the softmax. What is the output at position $t$ used for? It predicts the token at position $t+1$.

## Q2. What does weight tying do, and what does it cost?

**60-second answer.** The LM head reuses the token embedding table, transposed. The same vectors map
ids in and scores out. It saves $V \cdot D$ parameters, and many GPT-style models use it. For $V = 50{,}257$ and
$D = 768$, that is about 38.6 million parameters. The cost is that the input
and output geometry are forced to match, and a model that needs them to differ may do better untied.

**Follow-ups.** Does tying change the number of logits? No. Only the parameter count and the coupling
change. Is there a bias on the head? In the usual GPT-style design, no.

## Q3. Why a final LayerNorm?

**60-second answer.** A pre-norm stream is not normalised inside the residual path. Each block adds to
it, so its scale grows with depth. The final LayerNorm gives the head a stable input scale, which keeps
the logit scale predictable.

**Follow-ups.** Could the head normalise instead? It could, but the stream would still carry the growth
into the next step. Is it always needed? Many pre-norm designs use it, so treat it as the default.

## Common mistakes

- Transposing the wrong axis for the tied head, which still has shape $(B, T, V)$ when $V = D$.
- Adding the position row by batch index instead of by position $t$.
- Forgetting the final LayerNorm and comparing logits to a model that has it.
- Counting the head as an extra $V \cdot D$ when it is tied.
