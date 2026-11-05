---
title: "Transformer block"
minutes: 12
min_viable: "Read The system and Worked example, then do p01 using Hint 1 if needed."
---

## The system

The block keeps a **residual stream** $x\in\mathbb{R}^{B\times T\times D}$: $B$ is the batch, $T$ the tokens, $D$ the model width. Attention uses $H$ heads of width $d_\text{head}=D/H$. The feed-forward sub-layer expands to width $F$. Projections use $y = xW + b$ with $W$ stored as (in, out). Every step below names its output shape.

1. $h = \mathrm{LN}_1(x)$. Shape $(B,T,D)$.
2. $q,\,k,\,v = hW_Q+b_Q,\ hW_K+b_K,\ hW_V+b_V$. Each is $(B,T,D)$. Reshape to $(B,T,H,d_\text{head})$, then swap to $(B,H,T,d_\text{head})$.
3. $s = qk^\top/\sqrt{d_\text{head}}$. Shape $(B,H,T,T)$. Set $s_{ij}=-\infty$ for every $j>i$. Softmax over the last axis (keys).
4. $o = \mathrm{softmax}(s)\,v$. Shape $(B,H,T,d_\text{head})$. Swap back and reshape to $(B,T,D)$. Then $a = oW_O+b_O$, shape $(B,T,D)$.
5. $x' = x + a$. Shape $(B,T,D)$. This is the raw stream, not normalised.
6. $h' = \mathrm{LN}_2(x')$. Shape $(B,T,D)$.
7. $g = \mathrm{gelu}(h'W_1+b_1)$, shape $(B,T,F)$. Then $f = gW_2+b_2$, shape $(B,T,D)$.
8. $y = x' + f$. Shape $(B,T,D)$.

Rules the shapes do not show:

| Rule | Statement |
|---|---|
| LayerNorm placement | $\mathrm{LN}$ wraps each sub-layer **input**, never $x+\text{sub-layer}(x)$ |
| LayerNorm definition | Over the last axis, with **biased** variance $\sigma^2$ (divide by $D$) and $\varepsilon=10^{-5}$. $\gamma,\beta\in\mathbb{R}^D$ are learned |
| Mask | Acts on scores, before the softmax. The diagonal $j=i$ stays visible |
| GELU | Exact form, $\mathrm{gelu}(z)=\tfrac12 z\,(1+\mathrm{erf}(z/\sqrt2))$ |

## Worked example

Take $B=1$, $T=2$, $D=2$, one head, $d_\text{head}=2$, and $W_Q=W_K=W_V=W_O=W_1=I$. Set $W_2=0$ and all biases to 0. Both LayerNorms have $\gamma=1$, $\beta=0$. Let $x = [[1,0],[0,1]]$.

Sub-layer 1. Normalising row 0 gives $\mathrm{LN}([1,0]) \approx [1,-1]$, because $\mu=0.5$ and $\sigma^2=0.25$. Row 0 may attend only to itself, so its attention output is its own value, $[1,-1]$. The stream becomes $x'_0 = [1,0]+[1,-1] = [2,-1]$.

Row 1 is $[-1,1]$ after $\mathrm{LN}_1$. It attends to both tokens. Its scores are $q_1\cdot k_0/\sqrt2 \approx -1.414$ and $q_1\cdot k_1/\sqrt2 \approx +1.414$. The softmax weights are about $0.056$ and $0.944$. The output is $0.056\,[1,-1] + 0.944\,[-1,1] \approx [-0.888, 0.888]$. So $x'_1 = [0,1]+[-0.888,0.888] = [-0.888, 1.888]$.

Sub-layer 2 adds zero, because $W_2=0$ and the FFN bias is zero. So $y = x'$, which matches the problem's worked example.

## Why it works

With $\mathrm{LN}$ on each branch input, the stream has an identity path. For $y = x + f(\mathrm{LN}(x))$, $\partial y/\partial x = I + \partial f/\partial x$, so the upstream gradient reaches the stream with an unscaled copy. Post-norm puts a LayerNorm on the stream itself, at every block.

Xiong et al. (2020) analyse this placement at initialisation. Their argument is that pre-norm trains stably and is reported to work without a learning-rate warm-up. The original post-norm Transformer used a warm-up schedule (Vaswani et al. 2017). Treat the analysis as motivation, not an exact law. The cost of pre-norm is that the output stream is not normalised, so many implementations add one final LayerNorm before the output head.

The causal mask is one line: set scores where $j>i$ to $-\infty$ before the softmax. This is the only thing that stops next-token training from copying the answer.

## Signal vs. noise

The signal is where the two LayerNorms and two residual additions sit, and that the mask acts on scores, not outputs. Embeddings, the output head and dropout are real parts of a model but not of this block. Ignore them for now.

Parameter count, with biases and LayerNorm:

$$P_\text{block} = \underbrace{4D^2 + 4D}_{\text{attention}} + \underbrace{4D}_{\text{two LN}} + \underbrace{2DF + F + D}_{\text{FFN}} = 4D^2 + 2DF + 9D + F.$$

For $D=512$ and $F=2048$ this is about $3.15$ million per block. The $4D^2$ attention term and the $2DF$ FFN term dominate. Biases and LayerNorm add only a few thousand parameters.

## Where it breaks

Three bugs produce plausible output. Post-norm placement trains, but usually needs warm-up. A missing or misaligned mask leaks future tokens, and the loss looks too good because the model sees the answer. A missing residual turns the block into a plain feed-forward stack, so deep blocks lose the stream.

Two numerical bugs pass a shape check. The tanh GELU approximation differs from the exact form by up to about $5\times10^{-4}$. Dividing the variance by $D-1$ rescales every normalised value by $\sqrt{(D-1)/D}$.

A mask that includes the diagonal leaves the first row with every score at $-\infty$. Softmax then returns NaN for that row. Always keep $j=i$ visible.

## Interview angle

Start with the stream: "the block reads $x$, adds two sub-layer outputs, and each sub-layer sees a normalised copy". Then name the placement, the mask and the scale $1/\sqrt{d_\text{head}}$. A likely follow-up asks why pre-norm trains more stably. Give the Xiong et al. answer: the residual path is an identity, so gradients reach the stream unscaled. A second follow-up asks how many parameters a block has. Use $4D^2 + 2DF + 9D + F$, and say the attention term dominates.

## Connections

This block combines dense layers, softmax over scores, and residual additions. It builds on:

- [`dl/dense-layer-forward-pass`](../dense-layer-forward-pass/lesson.md) for the projections
- [`cml/softmax`](../../cml/softmax/lesson.md) for the attention weights
- [`dl/layer-normalization`](../layer-normalization/lesson.md) for both LayerNorms
- [`dl/residual-connections`](../residual-connections/lesson.md) for the stream and its identity path
- [`dl/causal-masking`](../causal-masking/lesson.md) for the mask
- [`dl/multi-head-attention`](../multi-head-attention/lesson.md) for the attention sub-layer
- [`dl/position-wise-feed-forward-block`](../position-wise-feed-forward-block/lesson.md) for the FFN sub-layer

The full model stacks these blocks $L$ times, with an embedding in front and an output head behind. Sources are listed in `topic.yaml`.
