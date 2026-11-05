---
title: "RMSNorm"
minutes: 10
min_viable: "Read The system and Worked example, then do p01 using Hint 1 if stuck."
---

## The system

RMSNorm keeps the scaling of layer normalization and drops the centering. Let
$x\in\mathbb{R}^{\cdots\times D}$ hold the features of one example along the last
dimension, and let $g\in\mathbb{R}^{D}$ be a learned gain. The rule:

| Quantity | Definition | Shape |
|---|---|---|
| Mean square | $\bar{x^2}=\frac{1}{D}\sum_j x_{\dots j}^2$ | leading dims |
| Root mean square | $\mathrm{RMS}(x)=\sqrt{\bar{x^2}+\epsilon}$, eps inside the root | leading dims |
| Output | $y=g\odot\frac{x}{\mathrm{RMS}(x)}$ | same as $x$ |

Compare with [layer normalization](../layer-normalization/lesson.md): the mean
$\mu$ and the subtraction $x-\mu$ are gone. There is no bias term in the standard
form, and the only learned parameter is $g$.

## Worked example

Take $x=[1,3]$, so $D=2$, with $g=[1,1]$ and $\epsilon$ negligible. The mean
square is $\frac{1+9}{2}=5$, so $\mathrm{RMS}=\sqrt5\approx2.236$. The output is
$y=[0.447,\,1.342]$.

Layer norm on the same input centers first. The mean is $2$, the deviations are
$[-1,1]$, and the output is $[-1,1]$. The outputs differ because RMSNorm leaves the
mean in place. Under a constant offset $c$ added to every feature, layer norm returns
the same output, but RMSNorm does not.

## Why it works

The question the paper asks is whether layer normalization needs its re-centering
step at all. Zhang and Sennrich (2019) propose a version that keeps only the
rescaling [arXiv:1910.07467]. The part that follows from the definition is the cost:
removing the mean removes one reduction and one broadcast subtraction per row, and
every normalization in the network pays that cost. The paper is not in the local
corpus, so this lesson makes no claim about its experimental results. Check the
paper before you quote a figure.

## Signal vs. noise

Signal: the formula, the placement of eps inside the root, and the absence of a bias.

Noise: the exact eps value (small defaults are common), and the gain's initial
value. What matters in practice is that RMSNorm and layer norm give different
outputs for the same input, so do not swap one for the other in a checkpoint without
retraining.

## Where it breaks

- **eps outside the root.** Inputs whose root mean square is near $\sqrt{\epsilon}$
  are scaled to a different magnitude. A test with small magnitudes catches this.
- **Adding a mean subtraction.** This silently turns the function into layer norm
  and breaks any checkpoint trained with RMSNorm.
- **Very large inputs.** Squaring can overflow when the values are close to the
  float range. A common fix is to divide by the largest magnitude before squaring.

## Interview angle

Sixty-second answer: RMSNorm divides each example by its root mean square and
applies a gain. It drops the mean subtraction and the bias, which saves a reduction
per row and keeps scale invariance. It is the normalizer used by many modern
decoder-only language models. The trade-off is the loss of shift invariance, and
the honest claim is that it is a cheaper design choice that many modern models use, and I would
check the original paper before I quote a comparison. Follow-ups are in `interview.md`.

## Connections

Prerequisite: [layer normalization](../layer-normalization/lesson.md), where the
per-example reduction, the eps placement and the gain are introduced. The
historical lineage runs from batch normalization to layer normalization to RMSNorm.
Each step changes what is normalized: first the batch axis, then the centering. Zhang and
Sennrich (2019) introduced RMSNorm [arXiv:1910.07467]. LLaMA uses it before each sublayer
[library:AI-ML/13-Scaling-and-Alignment/Touvron et al. — LLaMA 2023.md].
