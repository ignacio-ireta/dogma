---
title: 'Smooth activations: tanh, GELU, Swish'
minutes: 12
min_viable: Read The system and Worked example, then implement p01.
---

## The system

A smooth activation can squash an input or smoothly gate it. For real scalar $x$, let $s(x)=1/(1+e^{-x})$ be the sigmoid, $\Phi(x)$ the standard normal cumulative distribution function, and $\phi(x)=e^{-x^2/2}/\sqrt{2\pi}$ its density. Apply these scalar rules elementwise to an array $X$ of any shape.

| Activation | Value | Local derivative |
|---|---|---|
| tanh | $\tanh(x)$ | $1-\tanh^2(x)$ |
| exact GELU | $x\Phi(x)$ | $\Phi(x)+x\phi(x)$ |
| Swish, beta 1 | $xs(x)$ | $s(x)+xs(x)(1-s(x))$ |

P01 implements values; optional p02 implements derivatives. These are three-item tuples in table order. This topic fixes exact GELU and beta 1; named approximations and other beta values are separate conventions. Start by predicting every value and derivative at zero.

## Worked example

At $x=0$, all three values are zero. Their derivatives are respectively $1$, $0.5$ and $0.5$. The gates pass zero as a value, but their slopes describe how the output responds to a small perturbation around zero.

At $x=1$, $\tanh(1)\approx0.761594$, $\Phi(1)\approx0.841345$, and $s(1)\approx0.731059$. The activation values are those same numbers because the input is one. The GELU derivative adds $1\cdot\phi(1)\approx0.241971$, giving $1.083315$. A gate's derivative need not equal its gate value, and need not stay below one.

## Why it works

The product rule explains both gated derivatives: differentiating $xg(x)$ gives $g(x)+xg'(x)$. Dropping the second term is the central conceptual mistake in p02. For GELU, $g'=\phi$; for Swish, $g'=s(1-s)$.

[Hendrycks and Gimpel](https://arxiv.org/abs/1606.08415) define GELU through the normal CDF. The [Swish paper](https://arxiv.org/abs/1710.05941) studies the sigmoid-gated family. Smoothness removes a kink; it does not make all derivatives large or positive. Tanh still saturates, and the gated activations can have a small region of negative slope.

## Signal vs. noise

Focus on definitions, shape preservation and the product rule. Memorizing decimal values is unnecessary; zero and tail behavior are more useful reconstruction anchors. Be explicit about exact versus approximate GELU because both are commonly implemented and have slightly different outputs and derivatives.

The exercise uses float64 and a bounded input domain. It is intended for rebuilding the formulas, not benchmarking accelerator kernels. Efficient production code may use fused operations or approximations chosen for hardware. Such changes need their own accuracy contract.

## Where it breaks

The naive sigmoid expression computes $e^{-x}$, which can overflow for a large negative input. Compute $q=e^{-|x|}$ instead: use $1/(1+q)$ on the nonnegative side and $q/(1+q)$ on the negative side. Both expressions are safe even if an array selection evaluates both branches.

The domain here is finite real values in $[-1000,1000]$. Tails can round to their limiting values, and tiny densities can underflow to zero. Preserve scalar and empty shapes. Complex and out-of-range inputs are rejected. Compare derivatives with central differences of independently written forward formulas, rather than with another rearrangement of the same backward formula.

## Interview angle

Lead with the function's role and convention: tanh squashes to a bounded signed range, while GELU and Swish weight inputs by smooth gates. Derive one product-rule derivative aloud. Follow-ups usually concern tanh saturation and the difference between GELU definitions. State that smoothness alone does not ensure better training for every architecture or dataset.

## Connections

ReLU supplies the hard-gate comparison. These local derivatives become multipliers in scalar backprop and dense-network training. Initialization remains coupled to the activation's signal behavior; see the [local initialization note](../../../sources/library/AI-ML/10-Backpropagation/Glorot%20Bengio%20Initialization%202010.md). End by writing which convention and tuple order your implementation uses, so the next session starts from a fixed interface.
