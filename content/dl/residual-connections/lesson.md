---
title: "Residual connections"
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 using Hint 1 if needed."
---

## The system

A residual block computes its output as a shortcut plus a learned branch:

$$y = S(x) + F(x)$$

Here $x\in\mathbb{R}^{d_{in}}$ is one input row, $F:\mathbb{R}^{d_{in}}\to\mathbb{R}^{d_{out}}$ is the residual branch (for example dense, ReLU, dense), and $S$ is the shortcut. Batches add leading axes; the feature axis is always last.

| Case | Shortcut $S(x)$ | Requirement |
|---|---|---|
| Identity | $S(x)=x$ | $d_{out}=d_{in}$ |
| Projection | $S(x)=xW_s$, with $W_s\in\mathbb{R}^{d_{in}\times d_{out}}$ | none; used when widths differ |

Use the identity whenever the widths agree. A projection is a learned linear map; with no bias in the shortcut, it adds $d_{in}\cdot d_{out}$ parameters. The shape rule is the whole contract: $y$ must have the same shape as $S(x)$, so a mismatched shortcut is an error, not a silent crop or pad.

## Worked example

Start with a scalar block $F(x)=0.5\,x$ and input $x=2$. The shortcut is the identity, so $y=2+0.5\cdot2=3$. The local derivative is $dy/dx = 1 + dF/dx = 1 + 0.5 = 1.5$. The identity contributes the $1$; the branch contributes $0.5$.

Now stack ten such blocks, each with local branch derivative $a = dF/dx = 0.1$. Residual stacks multiply factors $(1+a)$, so the total gain is $1.1^{10}\approx 2.59$. A plain stack multiplies $a$ itself, so the gain is $0.1^{10}=10^{-10}$. The residual gradient neither vanishes nor explodes here; the plain gradient has disappeared.

For a projection, take $x=(1,-2)$, $W_s=(0.5,0.5)^T$ and a branch that outputs $(1)$. The shortcut is $x W_s = 1\cdot0.5 + (-2)\cdot0.5 = -0.5$, so $y = -0.5 + 1 = 0.5$.

## Why it works

**Degradation, not overfitting.** He et al. observed that a deeper plain network can have *higher training error* than a shallower one (their Fig. 1; corpus: 1512.03385, §1). A deeper network can always copy a shallower solution by setting the extra layers to identity, so the optimiser, not the capacity, is failing. The paper's fix is to make the identity the default: the block learns $F$, the residual, and an identity mapping needs only $F \to 0$.

**The gradient identity.** Differentiate the block. Because $S$ is the identity,

$$\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$$

where $I$ is the $d_{in}\times d_{in}$ identity matrix and $\partial F/\partial x$ is the branch Jacobian. The $I$ term is a path with gain 1 that bypasses the branch. Stacking $L$ blocks gives products of such terms, so the gradient reaches earlier layers through the identity path even when every branch Jacobian is small. This is why the example gave $1.1^{10}$ and not $0.1^{10}$. The claim is limited: the identity term guarantees one unit-gain path per block, not a bounded gradient overall, since the branch terms can still cancel or grow.

## Signal vs. noise

Signal: the shortcut is the identity, the branch is the learned part, and the shapes of $S(x)$ and $F(x)$ must match. Noise: the exact widths and the number of blocks, which vary by architecture. Ignore the convolutional case for now; it changes the tensor layout, not the rule.

## Where it breaks

Mismatched widths are the usual bug. An identity shortcut with $d_{out}\neq d_{in}$ has no valid meaning; reject it, and use a projection. Second, applying the activation after the sum, $\mathrm{ReLU}(x+F(x))$, changes the identity path into a nonlinear one and removes the clean gradient identity. Third, detaching the shortcut (`x.detach()`) gives the right output but kills the identity gradient, so the forward pass looks correct and the training curve is slow. Check the input gradient against an independent computation.

**Pre-norm vs. post-norm in transformers.** With a normalisation $N$ (layer normalisation, covered in its own topic):

- Post-norm: $x_{l+1} = N(x_l + F(x_l))$. The normalisation sits on the identity path, so the signal passes through $N$ at every block.
- Pre-norm: $x_{l+1} = x_l + F(N(x_l))$. The identity path carries $x_l$ unchanged; $N$ only feeds the branch.

The original Transformer, which is post-norm, trains with a learning-rate warm-up schedule (Vaswani et al. 2017, eq. 3). Xiong et al. (2020, arXiv:2002.04745) analyse the placement and argue that pre-norm is easier to train without that warm-up. That paper is not in the local corpus, so its specific claims are unverified here. Pre-norm is also often reported to reach slightly weaker final performance than a well-tuned post-norm model; check the source before quoting it. Name the choice and its cost; do not call one ordering universally correct.

## Interview angle

Sixty-second answer: a residual block learns a correction $F$ on top of an identity shortcut, so deep stacks can represent identity cheaply and gradients have a unit-gain path. Shapes must match, or the shortcut needs a projection. Transformers use pre-norm for stability at the cost of a slightly different final representation. Likely follow-up: why does the identity help the gradient? Answer with $\partial y/\partial x = I + \partial F/\partial x$. Second follow-up: when do you need a projection? When the width changes, including $d_{in}\neq d_{out}$ in an attention output projection.

## Connections

Prerequisites: dense layer forward pass and backprop through a dense layer; the branch $F$ is a two-layer dense block and its backward pass is exactly the dense VJP. This topic leads to attention blocks, which wrap multi-head attention in the same residual pattern, and to the transformer block, where pre-norm and post-norm differ. The normalisation $N$ is [layer normalization](../layer-normalization/). The attention that fills the branch in a transformer is [scaled dot-product attention](../scaled-dot-product-attention/). Historical lineage: He et al. (2016), arXiv:1512.03385, introduced the residual formulation for CNNs; Xiong et al. (2020), arXiv:2002.04745, analysed normalisation placement in transformers.
