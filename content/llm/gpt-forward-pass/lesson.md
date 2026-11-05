---
title: "GPT forward pass"
minutes: 12
min_viable: "Read The system and the shape table, then do p01 with Hint 2 open."
---

## The system

A decoder-only language model maps a block of token ids to one score per vocabulary entry, at
every position. Call the batch size $B$, the number of tokens $T$, the vocabulary size $V$, the
model width $D$, the number of blocks $N$, the number of heads $H$ with head width
$d_\text{head}=D/H$, and the context limit $C$ (the number of rows in the position table).

The forward pass has five steps. Every row of the table names the output shape.

| Step | Operation | Output shape |
|---|---|---|
| 1. Input | integer ids `idx`, with entries in $[0,V)$ | $(B,T)$ |
| 2. Embed | $x_{b,t}=\mathrm{tok\_emb}[idx_{b,t}]+\mathrm{pos\_emb}[t]$ | $(B,T,D)$ |
| 3. Blocks | $x \leftarrow \mathrm{Block}_n(x)$ for $n=1,\dots,N$, pre-norm and causal | $(B,T,D)$ |
| 4. Final LayerNorm | $h=\mathrm{LN}_f(x)$ | $(B,T,D)$ |
| 5. LM head | $\text{logits}=h\,W_\text{head}$ with $W_\text{head}\in\mathbb{R}^{D\times V}$ | $(B,T,V)$ |

Weights follow the row-vector convention of `dl/transformer-block`: $y = xW + b$, with $W$ stored
as (in, out). The embedding table $\mathrm{tok\_emb}$ is $(V,D)$, one row per token. The position
table $\mathrm{pos\_emb}$ is $(C,D)$, one row per position.

**Tying.** The head has two options, and one rule decides which:

| Head | Matrix used | Extra parameters |
|---|---|---|
| Tied | $W_\text{head}=\mathrm{tok\_emb}^\top$, shape $(D,V)$ | none |
| Untied | a separate $W_\text{head}$, shape $(D,V)$ | $D\cdot V$ |

With tying, the logit for token $v$ at position $(b,t)$ is $h_{b,t}\cdot\mathrm{tok\_emb}[v]$. The
input lookup and the output score use the same vectors.

## Worked example

Take $V=3$, $D=2$, $T=1$, $N=0$ blocks, and a tied head. Let

$$\mathrm{tok\_emb}=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix},\qquad \mathrm{pos\_emb}[0]=(0,0),\qquad \gamma_f=(1,1),\ \beta_f=(0,0),$$

and input `idx = [[1]]`.

1. Embed. Row 1 of $\mathrm{tok\_emb}$ is $(0,1)$. Add $\mathrm{pos\_emb}[0]=(0,0)$. So $x=(0,1)$, shape $(1,1,2)$.
2. Blocks. There are none, so $x$ is unchanged.
3. Final LayerNorm. The mean is $0.5$ and the biased variance is $0.25$. So $h \approx (-1,\,1)$. The epsilon makes it $(-0.99998,\ 0.99998)$.
4. Tied head. $W_\text{head}=\mathrm{tok\_emb}^\top$, shape $(2,3)$. The logits are
   $h\,W_\text{head} = (-1\cdot1+1\cdot0,\ -1\cdot0+1\cdot1,\ -1\cdot1+1\cdot1) = (-1,\ 1,\ 0)$.

The output has shape $(1,1,3)$, one score per vocabulary entry. Token 1 scores highest, because
$h$ points along its own row. Now add one zero block. Its residual adds zero, so the logits do not
change. That confirms the residual path.

## Why it works

Each stage has one job. The embedding turns ids into vectors, and the position table gives the
model order, since attention alone cannot see position [corpus: library-papers/1706.03762.pdf.md].
The blocks refine the stream. The final LayerNorm is needed because the pre-norm stream is never
normalised inside the residual path, so the head would otherwise see an unnormalised scale. The
head turns the stream into scores.

Press and Wolf (2017) proposed using the output embedding in language models, which is the tied
head: one matrix serves as both the input table and the output projection. Many GPT-style models use
the tied form. The parameter saving is exact. For $V=50{,}257$ and $D=768$ (the GPT-2 vocabulary and
width in the GPT-2 paper), the untied head would add $V\cdot D = 38.6$ million parameters. The number
of logits is the same either way.

## Signal vs. noise

Signal: the five stages, the $(B,T,V)$ output, the tie rule, and the rule that the position table
is added per position, not per batch row. Noise: the exact epsilon, initialisation schemes, and
dropout, which this forward pass omits. Serving-time caching of keys and values is a different topic.

## Where it breaks

- **Transposed head.** With $V = D$, a head written as `h @ tok_emb` has the right shape and the
  wrong values. Check that the tied head is `tok_emb.T`.
- **Position added per row.** Adding `pos_emb[0]` to every position makes identical tokens look the
  same. The position table must be indexed by $t$.
- **Context overflow.** A sequence with $T > C$ has no position row for its tail. Fail loudly.
- **Negative ids.** Python and PyTorch will index from the end, so $-1$ silently reads the last token.
  Reject ids outside $[0,V)$ before the lookup.
- **Missing final LayerNorm.** The model still runs, but its logit scale depends on the depth of the
  stream.

## Interview angle

Start with the stream: "embed, run $N$ pre-norm causal blocks, normalise, project to $V$". Then say the
shapes: $(B,T)$ in, $(B,T,V)$ out. The likely follow-ups are "what does weight tying buy?" and "why a final
LayerNorm?". For tying, give the exact saving: $V\cdot D$ parameters, and the same vectors serving as input
and output. For the final LayerNorm, say that the pre-norm stream is not normalised. The head then sees a
stable scale.

## Connections

This topic puts the parts together. It builds on:

- [`dl/transformer-block`](../../dl/transformer-block/lesson.md) for the block, its 16 parameter keys and
  the pre-norm placement
- [`dl/positional-encodings`](../../dl/positional-encodings/lesson.md) for the position table
- [`dl/layer-normalization`](../../dl/layer-normalization/lesson.md) for the final LayerNorm

It unlocks parameter counting in `llm/model-parameter-and-memory-footprint`, where the embedding and
head terms are counted, and the generation loop, which calls this forward pass once per new token.
The historical line runs from the GPT-2 paper to the decoder-only models that followed it
[library: AI-ML/12-Frontiers/Radford GPT-2 2019.md].
