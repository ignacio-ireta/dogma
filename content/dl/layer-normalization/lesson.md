---
title: "Layer normalization"
minutes: 12
min_viable: "Read The system and Worked example, then do the forward pass in p01 using Hint 1 if stuck."
---

## The system

Layer normalization standardizes each example across its own features. Let
$x\in\mathbb{R}^{\cdots\times D}$ hold any number of leading positions, where
each length-$D$ vector is one example's features at one position. Let
$\gamma,\beta\in\mathbb{R}^{D}$ be a learned gain and bias. The rule:

| Quantity | Definition | Shape |
|---|---|---|
| Mean | $\mu=\frac{1}{D}\sum_{j} x_{\dots j}$ | leading dims |
| Variance | $\sigma^2=\frac{1}{D}\sum_{j}(x_{\dots j}-\mu)^2$ (divide by $D$) | leading dims |
| Output | $y=\gamma\odot\hat x+\beta$, $\hat x=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}$ | same as $x$ |

Statistics run over the last dimension only. Nothing is shared across examples
and nothing is stored between calls. The same computation runs in training and
inference.

Two things to remember:

- **No running statistics.** Batch normalization keeps running means and
  variances because inference must not depend on the other examples in a batch.
  Layer normalization never needs them.
- **Eps inside the root.** $\epsilon>0$ sits inside the square root, with the
  variance. The same convention appears in RMSNorm next.

## Worked example

Take one example with features $x=[1,2,3]$, so $D=3$, with $\gamma=[1,1,1]$,
$\beta=[0,0,0]$, and $\epsilon$ negligible.

Mean $\mu=2$. Deviations are $[-1,0,1]$. Variance $\sigma^2=\frac{2}{3}\approx0.667$.
Normalized output $\hat x=[-1.225,\,0,\,1.225]$.

Now a batch with two rows, $x=\begin{bmatrix}1&2&3\\10&10&10\end{bmatrix}$. Row 1
behaves as above. Row 2 is constant, so its deviations are zero, and its output
is $\beta$ whatever the gain. This is the degenerate case: zero variance leaves
only $\epsilon$ in the denominator, and the centred numerator is also zero.

Per-feature affine parameters broadcast over the leading dims. With
$\gamma=[2,1,0.5]$ and $\beta=[0,0,1]$, row 1 becomes
$y=[-2.449,\,0,\,1.612]$. The gain and bias sit on the last axis, not on the batch.

## Why it works

The statistics answer a different question from batch normalization. Batch
normalization asks how one feature varies across examples. Layer normalization
asks how the features of one example vary among themselves. The paper motivates
this with recurrent networks, where batch statistics are hard to apply
[library-papers/1607.06450 p.1]. It also stresses that the computation
is identical at training and test time [library-papers/1607.06450 p.1].

The numerical rule that matters is centering before squaring. The variance is
computed from $x-\mu$, not from $\mathbb{E}[x^2]-\mu^2$. When the features share
a large offset, the one-pass form loses most of its significant digits, because
$\mathbb{E}[x^2]$ and $\mu^2$ are both huge and nearly equal.

## Signal vs. noise

Signal: the normalization axis, the absence of running statistics, and the biased
variance over $D$.

Noise: the default eps, and the initial gain and bias (usually $1$ and $0$). Pre-norm
versus post-norm placement is a block-design choice. This topic only needs its
one-sentence description: pre-norm normalizes the input of each sublayer,
post-norm normalizes after the residual add. The original transformer used
post-norm [library-papers/1706.03762 p.3]. Residual connections come next week,
and the placement only becomes meaningful there.

## Where it breaks

- **Normalizing over the wrong axis.** Reducing over the batch turns the layer
  into batch normalization. The batch mean then depends on other examples.
- **Constant features.** A row with zero variance returns the bias. This is correct
  behaviour; it is not a bug to fix with a larger eps.
- **Large offsets.** The one-pass variance formula fails silently when inputs share
  a large mean. The test for this uses an offset of $10^6$.
- **Dimension one.** With $D=1$ every row is constant, so the output is the bias.
  This is valid but usually means the wrong axis was chosen.

## Interview angle

Sixty-second answer: layer normalization standardizes each example over its own
features, so it keeps no running statistics and computes the same thing at
training and inference. That is why it fits variable-length sequences and small
batches. The costs are a per-token reduction and the usual numerical care when
centering. Follow-ups are in `interview.md`: the difference from batch norm, and
where normalization sits in a transformer block.

## Connections

This builds on [batch normalization](../batch-normalization/lesson.md), which
introduces the mean, the biased variance and the affine transform in per-feature
form. It unlocks [RMSNorm](../rmsnorm/lesson.md), which removes the centering.
The original paper is Ba, Kiros and Hinton (2016), which introduces it as a
per-example alternative to batch statistics [library-papers/1607.06450 p.2].
