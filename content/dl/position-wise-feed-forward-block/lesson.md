---
title: "Position-wise feed-forward block"
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 with the GELU formula beside you."
---

## The system

Let $X\in\mathbb{R}^{B\times L\times d}$ hold $B$ sequences of $L$ tokens with width $d=d_{model}$. The block has two linear maps and a nonlinearity $\sigma$ in between:

$$\mathrm{FFN}(x)=W_2\,\sigma(W_1 x+b_1)+b_2,\qquad W_1\in\mathbb{R}^{d_{ff}\times d},\ W_2\in\mathbb{R}^{d\times d_{ff}}.$$

The map is applied to each position $x\in\mathbb{R}^{d}$ separately. The leading axes $B$ and $L$ are not mixed. The hidden width $d_{ff}$ is usually $4d$.

| Variant | Hidden path | Matrices | Weights (no biases) |
|---|---|---|---|
| Plain, GELU | $\sigma=\mathrm{GELU}$ | $W_1, W_2$ | $2\,d\,d_{ff}$ |
| Gated, SwiGLU | $\mathrm{SiLU}(Wx)\odot(Vx)$ | $W, V, W_2$ | $3\,d\,d_{ff}$ |

The exact GELU is $\mathrm{GELU}(z)=z\,\Phi(z)=\tfrac{1}{2}z\,(1+\mathrm{erf}(z/\sqrt2))$, where $\Phi$ is the standard normal CDF. SiLU is $\mathrm{SiLU}(z)=z\,\sigma_{\mathrm{logistic}}(z)$.

## Worked example

Take $d=2$, $d_{ff}=2$, input $x=[1,0]$, $W_1=I$, $b_1=0$, $W_2=I$, $b_2=0$. Then $W_1x=[1,0]$ and $\mathrm{GELU}(1)=\Phi(1)\approx0.8413$, $\mathrm{GELU}(0)=0$. The output is $[0.8413,\ 0]$.

Now the gated version on $x=[1,2]$ with $W=I$, $V=\begin{pmatrix}1&1\\0&1\end{pmatrix}$ and $W_2=I$. The gate is $g=[1,2]$ and the up path is $u=[3,2]$. $\mathrm{SiLU}(1)=0.7311$ and $\mathrm{SiLU}(2)=1.7616$, so the output is $[0.7311\cdot3,\ 1.7616\cdot2]=[2.1932,\ 3.5232]$.

## Why it works

Vaswani et al. [library:AI-ML/12-Frontiers/Vaswani Attention Is All You Need 2017.md] describe the block as two linear transformations with a nonlinearity between them, "two convolutions with kernel size 1", and use $d_{ff}=2048$ for $d_{model}=512$. That inner width is the $4\times$ default used here.

GELU is a smooth alternative to ReLU that weights each input by how likely it is under a standard normal, so the reference uses the exact erf form [paper:Hendrycks and Gimpel 2016, arXiv:1606.08415]. Gated linear units multiply two projections, and one of them is passed through a sigmoid-like function. Shazeer (2020, arXiv:2002.05202, "GLU Variants Improve Transformer") tests this family of variants in Transformers. The title and arXiv id are confirmed by the reference list of the LLaMA report in the local corpus, which uses SwiGLU; the paper itself is not in the corpus. The 2/3 width keeps the total weight count equal: $3\,d\cdot\tfrac83 d=8d^2=2\,d\cdot 4d$.

## Signal vs. noise

Signal: the per-position definition, the $4\times$ default, the exact GELU formula, the gate and up roles in SwiGLU, and the parameter count. Noise: the choice between GELU and SiLU for a given model is an empirical decision; the rounding of $\tfrac83 d$ to a hardware-friendly multiple is an engineering detail. Do not spend time on activation checkpointing here.

## Where it breaks

- **Mixing positions.** Flattening $(L,d)$ into one vector before the linear layers gives the block a sequence-wide weight. The test is that changing one token leaves the others unchanged.
- **Wrong orientation.** Linear layers are stored as $(out,in)$. Applying $xW_1$ instead of $xW_1^\top$ fails silently when the matrices are square.
- **Tanh GELU.** The approximation $0.5z(1+\tanh(\sqrt{2/\pi}(z+0.044715z^3)))$ differs from the exact form by small amounts. Oracles compare against the exact form, so the approximation fails at tight tolerances.
- **Gate placement.** Applying the activation to the up path instead of the gate gives a different function that still runs.
- **Width after switching to a gated block.** Keeping $d_{ff}=4d$ with three matrices increases the block's parameters by half.

## Interview angle

Say that the block is per-position, expands by about four times, and carries most of the layer's weights. For the gated variant, name the three matrices and the 2/3 width. The two likely follow-ups are "why four times" and "what does the 2/3 keep fixed". Full answers are in `interview.md`.

## Connections

Prerequisites: [dense layer forward pass](../dense-layer-forward-pass/) for the two affine maps, and [smooth activations: tanh, GELU, Swish](../smooth-activations-tanh-gelu-swish/) for the exact GELU and SiLU. The block sits after [multi-head attention](../multi-head-attention/) inside a Transformer block, with residual connections around both. The lineage runs from the MLP of the 1980s to the 2017 Transformer's position-wise block and the 2020 gated variants.
