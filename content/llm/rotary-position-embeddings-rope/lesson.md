---
title: "Rotary position embeddings (RoPE)"
minutes: 12
min_viable: "Read The system and the layout table, then do p01 with Hint 2 open."
---

## The system

RoPE is applied inside each attention head, to the queries and keys only. Call the batch $B$, the heads
$H$, the tokens $T$ and the head width $d$ (even). A query or key tensor has shape $(B,H,T,d)$. Token
$t$ sits at absolute position $p_t=\mathrm{offset}+t$, where $\mathrm{offset}$ is the number of tokens
already in the KV cache.

RoPE rotates each feature pair by an angle that depends on the position. Pair $i$ has frequency
$\theta_i=10000^{-2i/d}$ for $i=0,\dots,d/2-1$. Its angle at token $t$ is $\phi_{t,i}=p_t\,\theta_i$.

There are two layouts. They pair the features differently, so they are different functions:

| Layout | Pair $i$ is | Used by |
|---|---|---|
| Interleaved (adjacent) | $(x_{2i},\,x_{2i+1})$ | the original RoFormer description |
| Half-split | $(x_i,\,x_{i+d/2})$ | GPT-NeoX and LLaMA style code (`rotate_half`) |

Both rotate each pair by the same formula. In half-split form, with $a=x[\dots,:d/2]$ and $b=x[\dots,d/2:]$:

$$\begin{aligned}
\text{out}[\dots,:d/2] &= a\cos\phi - b\sin\phi,\\
\text{out}[\dots,d/2:] &= a\sin\phi + b\cos\phi,
\end{aligned}$$

with $\cos\phi$ and $\sin\phi$ of shape $(T,d/2)$, broadcast over $B$ and $H$. Values $V$ are never
rotated. The score $q_m\cdot k_n$ after rotation depends only on the offset $m-n$, which is why the
positions must be shifted for q and k alike.

## Worked example

Take $d=4$, so there are two pairs with $\theta_0=1$ and $\theta_1=10000^{-1/2}=0.01$. Use token position
$p=1$ and the input $x=(1,0,0,0)$.

**Half-split.** Pair 0 is $(x_0,x_2)=(1,0)$ with angle $1$. So $a=1$, $b=0$, and the output is
$(\cos 1,\ 0,\ \sin 1,\ 0)=(0.5403,\ 0,\ 0.8415,\ 0)$. Pair 1 is $(x_1,x_3)=(0,0)$ and stays zero.

**Interleaved.** The same input in the adjacent layout is $y=(1,0,0,0)$. Pair 0 is $(y_0,y_1)=(1,0)$,
so the output is $(0.5403,\ 0.8415,\ 0,\ 0)$.

**Conversion.** The two layouts differ by a fixed permutation of the feature axis. For $d=4$ it is $\pi=(0,2,1,3)$; in general $\pi=(0,2,4,\dots,1,3,5,\dots)$. The half-split input is
$x=y[\pi]$, and the half-split output is the interleaved output indexed the same way:
$(0.5403,\ 0.8415,\ 0,\ 0)[\pi]=(0.5403,\ 0,\ 0.8415,\ 0)$. The result matches. A model trained
in one layout needs its weight columns, or its rotation, permuted once to match the other.

## Why it works

Attention scores are dot products. Rotating by $R_m$ on the query side and $R_n$ on the key side gives
$(R_m q)^\top(R_n k)=q^\top R_{n-m}k$, because $R_m^\top=R_{-m}$. So the score depends only on the offset. Su et al. (2021)
introduced the method [paper: Su et al. 2021, arXiv:2104.09864]. LLaMA adopts it at every layer
[corpus: library-papers/2302.13971.pdf.md, section 2.2]. The method rotates only q and k, because the
score is where relative position is needed. Values are the content that gets averaged.

For decoding, the cache stores each rotated key. A new token must be rotated at position $\mathrm{past\_len}$,
not 0, or its score against every cached key is off by the same mismatch.

Context extension by **linear position interpolation**, the method of Chen et al. (2023)
[paper: Chen et al. 2023, arXiv:2306.15595], divides every position by a scale $s\ge1$: $p_t\to p_t/s$.
Positions up to $sL$ map into the range $[0,L)$ the model trained on, so the angles stay in that range,
at the cost of finer spacing between neighbouring tokens. With a uniform scale, dividing every position
by $s$ and dividing every frequency $\theta_i$ by $s$ give the same angles, so the choice between them is
cosmetic. The base $10000$ is not changed. Changing the base changes the frequencies non-uniformly, which
is a different method, covered in a later topic.

## Signal vs. noise

Signal: the two layouts and their permutation, the cache offset, that q and k rotate and v does not, and
that interpolation divides the angle by $s$ with the base fixed. Noise: the exact base, which checkpoints fix.
The layout matters only when you load weights.

## Where it breaks

- **Layout mismatch.** Adjacent pairs fed to a half-split model run without error and give wrong scores.
  Check which layout the checkpoint was trained with before you load it.
- **Offset ignored in decoding.** Rotating the new token at 0 passes shape checks, and the output looks
  plausible. Compare the last row of a full pass with a one-token step.
- **Rotating values.** It is not an error in shape, but it changes what the attention output means.
- **Interpolation without fine-tuning.** Compressing positions makes neighbours sit closer together in
  angle than the model saw in training. Interpolation is usually paired with some training.

## Interview angle

Say that RoPE puts position into the score as an offset, applied to q and k inside each head. Follow-ups:
"why not v?" (rotation serves the score, and v is the content averaged), "how does the cache change the
positions?" (new tokens start at the number of cached tokens), and "what does interpolation cost?" (longer
positions fit the trained range, but neighbours are closer in angle).

## Connections

This topic builds on [`dl/positional-encodings`](../../dl/positional-encodings/lesson.md), whose p02 is
the interleaved-pair RoPE on a whole sequence, starting at position 0 and written with complex numbers.
This topic adds the half-split layout, a cache offset and a scale. It sits inside the heads of
[`dl/multi-head-attention`](../../dl/multi-head-attention/lesson.md). The offset rule is what the KV cache
depends on in the incremental decoding topic. Later topics cover base changes for context extension and
partial RoPE, which leaves some dimensions unrotated. LLaMA uses RoPE together with pre-norm blocks
[library: AI-ML/13-Scaling-and-Alignment/Touvron et al. — LLaMA 2023.md].
