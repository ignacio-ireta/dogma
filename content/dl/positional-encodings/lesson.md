---
title: "Positional encodings"
minutes: 12
min_viable: "Read The system and Worked example, then do p01 with the table in front of you."
---

## The system

Let a sequence have $L$ tokens, each with an embedding $x_t\in\mathbb{R}^{d}$ for position $t=0,\dots,L-1$. Attention computes $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$, where $Q,K,V$ are linear maps of the token rows. Each output row is a weighted sum of value rows, and the weights depend only on the content of the token pair.

Two ways to add position:

| Scheme | Where it enters | Parameters | Score depends on |
|---|---|---|---|
| Sinusoidal (absolute) | Added to $x_t$ once, before attention | none | $\mathrm{PE}_m$ and $\mathrm{PE}_n$ separately |
| Learned absolute | Added to $x_t$ once | $L_{max}\times d$ table | the two table rows separately |
| Rotary (RoPE) | Rotates $q_t$ and $k_t$ inside each head | none | the offset $m-n$ |

**Sinusoidal table.** For $pos\in\{0,\dots,L_{max}-1\}$ and pair index $i\in\{0,\dots,d/2-1\}$ (with $d$ even):

$$\mathrm{PE}[pos,2i]=\sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad \mathrm{PE}[pos,2i+1]=\cos\!\left(\frac{pos}{10000^{2i/d}}\right).$$

The table has shape $(L_{max}, d)$. Even columns are sines and odd columns are cosines, interleaved. The frequency of pair $i$ is $\omega_i=10000^{-2i/d}$, which falls geometrically from $1$ (pair 0) to about $10^{-4}$.

**RoPE.** Read each adjacent pair $(x_{2i},x_{2i+1})$ of a query or key row at position $t$ as a 2-D vector, and rotate it by angle $t\,\omega_i$. Values are not rotated.

## Worked example

Take $d=4$, so there are two pairs with $\omega_0=10000^0=1$ and $\omega_1=10000^{-1/2}=0.01$. For position $t=1$:

- pair 0 angle $=1\cdot 1=1$: entries $[\sin 1,\ \cos 1]=[0.8415,\ 0.5403]$.
- pair 1 angle $=1\cdot 0.01$: entries $[\sin 0.01,\ \cos 0.01]=[0.0100,\ 1.0000]$.

So row 1 is $[0.8415,\ 0.5403,\ 0.0100,\ 1.0000]$. Position 0 gives $[0,1,0,1]$ because every angle is zero.

For RoPE with $d=2$ and $t=1$, the vector $(1,0)$ becomes $(\cos 1,\ \sin 1)=(0.5403,\ 0.8415)$. The norm stays $1$. Now take $q$ at position $m$ and $k$ at position $n$. Their dot product after rotation is $\|q\|\|k\|\cos\big((n-m)\omega + \phi\big)$, where $\phi=\beta-\alpha$ is the angle of $k$ minus the angle of $q$ before rotation. Check with $\alpha=0$, $\beta=0.5$, $\omega=1$, $m=0$, $n=2$: the rotated vectors sit at $0$ and $2.5$, so the score is $\cos 2.5$, which matches $\cos(2+0.5)$. Shifting both positions by the same amount changes nothing.

## Why it works

Attention is permutation-equivariant: if the rows of $X$ are permuted by $P$, the output rows are permuted by $P$ too, because $\mathrm{softmax}$ and the weighted sum act row by row on content. Without position information, "dog bites man" and "man bites dog" get the same set of representations. Vaswani et al. [library:AI-ML/12-Frontiers/Vaswani Attention Is All You Need 2017.md] add a fixed sine-cosine code. For any fixed offset $k$, $\mathrm{PE}_{pos+k}$ is a linear function of $\mathrm{PE}_{pos}$ on each pair. That makes relative shifts easy to represent. Their experiment found learned embeddings gave nearly the same quality. They chose the sinusoidal form because it may let the model extrapolate to longer sequences; that was a hypothesis they gave, not a result they measured.

RoPE (Su et al., 2021) takes the relative idea further. It applies a rotation $R_t$ to the pair so that $(R_m q)^\top (R_n k)=q^\top R_{n-m}k$. The score is then a function of the offset alone, and no position vector is added to the content.

## Signal vs. noise

Signal: the table formula, the even/odd interleave, the frequency schedule, and the fact that RoPE rotates queries and keys only. Noise: the base 10000 is a convention chosen by Vaswani et al., not a law; any base that spreads frequencies over a useful range works, and the exact value matters only when you compare against a checkpoint. Skip learned-embedding training details for this topic.

## Where it breaks

- **Odd width.** The interleaved table needs whole sin/cos pairs. An odd $d$ must raise an error, not drop a column.
- **Block layout.** Putting all sines first and all cosines after is a different function. It can run without error and still give the wrong table.
- **Split-half vs adjacent pairs.** Some libraries rotate $(x_{0..d/2}, x_{d/2..d})$ and others rotate adjacent pairs. Both are valid rotations, but a model trained with one layout gives wrong scores with the other.
- **Long context.** Sinusoidal and RoPE angles at offsets never seen in training are out of distribution. Context-extension methods rescale the frequencies; this topic does not cover them.

## Interview angle

Lead with the reason: attention alone cannot see order. Then state where position enters (added to content, or rotating queries and keys) and what that makes the score depend on. Trade-offs come first. Sinusoidal has no parameters; learned tables are capped at the trained length; RoPE makes relative offset the native quantity. The two likely follow-ups are "why sines and cosines" and "why only queries and keys". Full answers are in `interview.md`.

## Connections

Prerequisites: [vector operations](../../cml/vector-operations/) for dot products and norms. Rotations preserve norm, which is why RoPE leaves score scale alone. Next comes [scaled dot-product attention](../scaled-dot-product-attention/), where the rotated $Q$ and $K$ enter the score, and [multi-head attention](../multi-head-attention/), where RoPE is applied inside each head. The lineage runs from the Fourier basis (sinusoids as a representation of position) to the 2017 Transformer and the 2021 rotary variant.
