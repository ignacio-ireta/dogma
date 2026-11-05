---
title: Xavier and He initialization
minutes: 12
min_viable: Read The system and Worked example, then implement p01.
---

## The system

Initialization sets both a starting scale and independent differences between units. Let $W\in\mathbb{R}^{d_{in}\times d_{out}}$ be a dense weight matrix. Here fan-in is $d_{in}$ and fan-out is $d_{out}$; they count input and output connections per unit. Choose a policy before drawing any values.

| Policy | Distribution | Target variance |
|---|---|---|
| Xavier uniform | $U(-a,a)$, $a=\sqrt{6/(d_{in}+d_{out})}$ | $2/(d_{in}+d_{out})$ |
| He normal | $N(0,s^2)$, $s=\sqrt{2/d_{in}}$ | $2/d_{in}$ |

Xavier balances the two fan directions under approximate linear signal assumptions. He adjusts for rectification. P01 fixes these exact policy variants and uses a fresh local seeded generator. First action: compute both scales for a layer with two inputs and six outputs.

## Worked example

For $d_{in}=2$ and $d_{out}=6$, Xavier gives $a=\sqrt{6/8}=\sqrt{0.75}\approx0.866025$. Uniform variance is $a^2/3=0.25$, agreeing with $2/(2+6)$. He gives $s=\sqrt{2/2}=1$, so its normal variance is one.

These are distribution properties. A tiny sampled matrix need not have sample mean zero or sample variance exactly equal to its target. Do not normalize each generated matrix afterward: that changes the specified sampling procedure. Repeating the same seed, shape and policy should reproduce the same matrix even after unrelated initialization calls.

## Why it works

For a preactivation $z=\sum_{i=1}^{d_{in}}w_i x_i$, independent zero-mean weights give the approximation $\operatorname{Var}(z)\approx d_{in}\operatorname{Var}(w)\operatorname{E}[x_i^2]$. Thus fan-in and weight variance jointly control signal scale. Backward propagation introduces a corresponding fan-out concern.

The [Glorot and Bengio note](../../../sources/library/AI-ML/10-Backpropagation/Glorot%20Bengio%20Initialization%202010.md) explains the Xavier compromise. For symmetric preactivation distributions, rectification removes half the second moment. The [He initialization note](../../../sources/library/AI-ML/10-Backpropagation/He%20Rectifier%20Initialization%202015.md) motivates compensating with variance $2/d_{in}$. This concerns second moments through a nonlinear gate; calling every quantity a variance after ReLU would obscure its positive mean.

## Signal vs. noise

The signal is activation choice, fan definition and the difference between variance and standard deviation. NumPy's normal `scale` is a standard deviation, so pass the square root. Uniform sampling takes bounds rather than a standard deviation. Weight orientation determines which stored dimension supplies each fan.

Skip framework gain options and specialized architectures today. This exercise initializes weights only; bias policy is separate. Zero biases do not by themselves create the symmetry problem of making every unit's entire incoming weight vector identical.

## Where it breaks

The derivation uses simplifying independence and distribution assumptions. Input correlations, depth, learned changes, normalization and skip connections can change actual signal behavior. Correct initialization supports training; it does not guarantee preserved gradients for every input or an eventual optimum.

Do not reset NumPy's global RNG inside a helper. A local generator makes the function independent of unrelated sampling and leaves callers' random streams intact. Reject invalid fans and seeds before drawing. Oversized allocations remain subject to available memory; this task does not promise to materialize arbitrary symbolic dimensions.

## Interview angle

State the nonlinearity and layout, then give the variance target and convert it to the requested distribution. For a ReLU dense layer in this convention, He-normal uses fan-in. If asked why a measured sample variance differs, distinguish a finite draw from its population target. If asked whether Xavier is wrong for every ReLU network, explain that these are informed starting rules rather than universal optimization guarantees.

## Connections

Parameter counting establishes the shape and storage budget. Dense forward reveals the sum whose scale initialization controls. Backprop shows why reverse signal scale also matters. The historical sequence runs from hidden-layer credit assignment to diagnosing depth and activation effects. After p01, record the chosen scheme, fan convention and seed as a reproducible starting-model contract.
