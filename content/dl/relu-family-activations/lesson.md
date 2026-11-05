---
title: ReLU-family activations
minutes: 12
min_viable: Read The system and Worked example, then implement p01.
---

## The system

An activation changes each element independently after a dense layer. For scalar input $x\in\mathbb{R}$ and fixed negative slope $a\in[0,1]$, define $f_a(x)=x$ when $x>0$ and $f_a(x)=ax$ otherwise. Ordinary ReLU uses $a=0$; leaky ReLU uses a positive slope. Arrays keep their shape because no axis is summed.

| Region | Forward value | Local derivative |
|---|---|---|
| $x<0$ | $ax$ | $a$ |
| $x=0$ | $0$ | $0$, our convention |
| $x>0$ | $x$ | $1$ |

The derivative at zero is an interface convention. When $a<1$, the ordinary two-sided derivative does not exist there. This exercise keeps zero even for $a=1$ so the branch rule is predictable. First action: fill the table for $a=0.1$ before opening p01.

## Worked example

For $x=[-2,0,3]$ and $a=0.1$, the values are $[-0.2,0,3]$ and the local derivatives are $[0.1,0,1]$. With ordinary ReLU, the values become $[0,0,3]$ and the derivatives $[0,0,1]$.

Let an upstream loss gradient be $g=[2,-4,5]$, with the same shape as $x$. The chain rule gives input gradient $g\odot f_a'(x)=[0.2,0,5]$, where $\odot$ means elementwise multiplication. P01 returns the local derivative only. Multiplying by $g$ belongs to the surrounding backward pass.

## Why it works

A straight line has constant slope within its region. When $a<1$, the rectifier joins lines with different slopes, making it nonlinear across the full domain while remaining simple locally. An activation that were affine everywhere would let a stack of dense layers collapse into one affine transformation. The sign-dependent rule prevents that collapse for $a<1$; at $a=1$, the forward map is the identity.

The [rectifier initialization note](../../../sources/library/AI-ML/10-Backpropagation/He%20Rectifier%20Initialization%202015.md) explains why negative-region behavior affects gradient propagation and initialization. A small negative slope provides a path for a gradient even when a unit's preactivation is negative. It does not guarantee that every unit will learn useful features.

## Signal vs. noise

The signal is the branch table and the difference between a local derivative and a loss gradient. Vectorization is secondary: once the scalar rule is clear, masks apply it to any array shape. Explicitly convert integer inputs so fractional negative outputs cannot be truncated.

Skip learnable slopes and in-place memory optimizations today. The slope in p01 is a fixed scalar shared across elements. PReLU is a related model where the slope itself is learned and therefore also needs a parameter gradient. That extension changes the parameter contract.

## Where it breaks

A unit that stays negative under ordinary ReLU receives no gradient through that activation. A large update or poorly scaled preactivation can leave it inactive across the training data. Leaky ReLU reduces this specific blockage, while other optimization problems may remain.

Do not use a symmetric finite difference exactly at zero to decide the chosen convention: it blends the two sides. Test points separated from zero by more than the difference step. Nonfinite values and invalid slopes are rejected by the exercise. Scalar and empty inputs still return NumPy arrays with exactly the original shape.

## Interview angle

Explain the trade-off in one sentence: the positive region has a simple unit slope, while ordinary ReLU blocks negative-region gradients; a leak retains a small negative-region path. Then state what happens at zero. Likely follow-ups are whether ReLU solves vanishing gradients everywhere and whether the leak is learned. Answer no to the first, and distinguish a fixed leaky slope from PReLU for the second.

## Connections

This is the nonlinear gate after the dense forward map. He initialization accounts for rectification when choosing initial weight scale. Smooth activations replace the kink with differentiable gates. The [local ReLU history note](../../../sources/library/AI-ML/10-Backpropagation/Nair%20Hinton%20ReLU%202010.md) gives the lineage from rectified units to deeper networks. End by recording whether any failure was a branch convention, dtype conversion or gradient-chain mistake.
