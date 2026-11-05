---
title: "Multi-head attention"
minutes: 12
min_viable: "Read The system and the shape table, then implement p01 with the table beside you."
---

## The system

Let $X\in\mathbb{R}^{B\times L\times d}$ hold $B$ sequences of $L$ tokens with width $d=d_{model}$. Let $h$ be the number of heads, and $d_k=d/h$ the width of each head. The layer has four weight matrices $W_q,W_k,W_v,W_o\in\mathbb{R}^{d\times d}$ and four bias vectors $b_q,b_k,b_v,b_o\in\mathbb{R}^{d}$, stored in the `nn.Linear` layout, so a projection is $y=xW^\top+b$.

The forward pass, with every shape named:

| Step | Operation | Shape after the step |
|---|---|---|
| 1 | $Q=XW_q^\top+b_q$, same for $K$ and $V$ | $(B,L,d)$ |
| 2 | reshape to $(B,L,h,d_k)$, then transpose axes 1 and 2 | $(B,h,L,d_k)$ |
| 3 | scores $S=QK^\top/\sqrt{d_k}$ over the last two axes | $(B,h,L,L)$ |
| 4 | $A=\mathrm{softmax}(S)$ over the key axis; context $C=AV$ | $(B,h,L,d_k)$ |
| 5 | transpose back to $(B,L,h,d_k)$, reshape to $(B,L,d)$ | $(B,L,d)$ |
| 6 | output $Y=CW_o^\top+b_o$ | $(B,L,d)$ |

Step 2 is the step to get right. The head axis moves in front of the token axis, so each $(L,d_k)$ matrix is one head. Step 5 is its inverse.

## Worked example

Take $L=2$, $d=2$, $h=1$ (so $d_k=2$), identity weights and zero biases. Let $x_0=[1,0]$ and $x_1=[0,1]$. Then $Q=K=V=X$, and $S=\begin{pmatrix}1&0\\0&1\end{pmatrix}/\sqrt{2}$. Row 0 of the softmax is $[e^{1/\sqrt2},1]/(e^{1/\sqrt2}+1)=[0.6698,\ 0.3302]$. So $Y_0=0.6698\,v_0+0.3302\,v_1=[0.6698,\ 0.3302]$.

Now set $h=2$ with $d=2$. Each head has width $d_k=1$, and the head split gives head 0 the feature 0 and head 1 the feature 1 of every token. Each head then attends over a single scalar feature. The output for each head is a separate one-feature attention, merged back into two features.

## Why it works

Each head computes one attention pattern: a row-stochastic matrix over keys. A single head with width $d$ can produce one such pattern per query. With $h$ heads, the model computes $h$ patterns from $h$ different projections, so it can attend to different relations at the same time, for example a neighbouring token and a syntactic head. Vaswani et al. [library:AI-ML/12-Frontiers/Vaswani Attention Is All You Need 2017.md] use $h=8$ with $d_k=d_v=d_{model}/h=64$, which keeps the total cost near that of one full-width head. The parameter count is $4(d^2+d)$ regardless of $h$, since $h$ only changes how the same $d\times d$ matrices are read. The $1/\sqrt{d_k}$ scale keeps the scores from growing with width.

## Signal vs. noise

Signal: the six steps, the shape table, the transpose in step 2, and the scale $\sqrt{d_k}$. Noise: the choice of $h$, which is a hyperparameter that matters less than the width $d$; and the initialization, which belongs in the initialization topic. Do not spend time on KV caching or efficient kernels here.

## Where it breaks

- **Reshape without transpose.** Reshaping $(B,L,d)$ directly to $(B,h,L,d_k)$ runs and returns the right shape, but each head then reads a slice that mixes tokens. The test is a reference layer with the same weights.
- **Scale by $\sqrt{d}$.** Using $\sqrt{d}$ instead of $\sqrt{d_k}$ shrinks the scores too much when $h>1$. The output looks plausible and loses sharpness.
- **Weight orientation.** A layer stored as $(out,in)$ needs $xW^\top$. Using $xW$ is silent when the matrices are square.
- **Mask convention.** PyTorch's `MultiheadAttention` treats `True` as blocked. This contract uses `True` as allowed, so the library call needs `~mask`.
- **Packed layout.** `torch.nn.MultiheadAttention` stores the three input projections as one `in_proj_weight` of shape $(3d,d)$: rows $0..d-1$ are $W_q$, rows $d..2d-1$ are $W_k$, and rows $2d..3d-1$ are $W_v$, each in $(out,in)$ order. `in_proj_bias` packs $b_q,b_k,b_v$ the same way, and `out_proj.weight` and `out_proj.bias` are $W_o$ and $b_o$. The oracle in the tests loads these fields, so a learner who reads the packed layout the other way round gets a mismatch.
- **Fully masked rows.** A row with no allowed key has an undefined softmax. Reject it rather than return NaN. This topic raises `ValueError`. The scaled dot-product and causal-masking topics return zeros or an all-`False` row instead, so state which rule the function uses.

## Interview angle

Lead with the shape table. Say that attention is computed per head on $d_k$ features, that the heads are concatenated, and that the output projection mixes them. The two likely follow-ups are "what do heads buy" and "how many parameters". The answer to the second is $4(d^2+d)$, which does not depend on $h$. Full versions are in `interview.md`.

## Connections

Prerequisites: [dense layer forward pass](../dense-layer-forward-pass/) for the projections, [softmax](../../cml/softmax/) for the weights, and [counting network parameters](../counting-network-parameters/) for the budget. Next comes the [position-wise feed-forward block](../position-wise-feed-forward-block/), which follows attention inside a Transformer block. The lineage is the Bahdanau-style additive attention of seq2seq models, generalised to scaled dot products and then to parallel heads in 2017.
