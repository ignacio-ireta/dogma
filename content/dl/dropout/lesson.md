---
title: "Dropout"
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 using Hint 1 if needed."
---

## The system

Dropout is a random transformation during training and an identity map during
evaluation. Inverted dropout scales surviving activations during training,
so evaluation needs no rescaling. Let $X\in\mathbb{R}^{n\times d}$ contain
activations and $G\in\mathbb{R}^{n\times d}$ contain the upstream derivative
of a scalar objective. The drop probability is $p\in[0,0.95]$ in this bounded
exercise; the keep probability is $q=1-p$.

Sample independent uniforms $U_{ij}$ and define the cached multiplier
$M_{ij}=\mathbf{1}[U_{ij}\ge p]/q$. Both $M$ and $U$ have shape $n\times d$.
The operation follows this decision table:

| Mode | Multiplier | Output | Input gradient |
|---|---|---|---|
| Training | Sampled $M$ | $X\odot M$ | $G\odot M$ |
| Evaluation | All ones | Copy of $X$ | Copy of $G$ |

Here $\odot$ means entrywise multiplication. The backward pass reuses the
forward multiplier. Our combined function returns the output, input gradient,
and multiplier. An explicit seed recreates the same mask on each call.
First action: identify which quantities change between train and evaluation.

## Worked example

Take $X=[[2,4]]$, $G=[[1,1]]$, $p=0.5$, and seed zero. The specified NumPy
generator's first uniforms are approximately $0.637$ and $0.270$.
The keep decisions are therefore $[1,0]$, and division by $q=0.5$ gives
$M=[[2,0]]$. The output is $[[4,0]]$ and the input gradient is $[[2,0]]$.
The dropped entry contributes neither an activation nor an input gradient.

Evaluation with the same inputs gives output $[[2,4]]$, gradient $[[1,1]]$,
and multiplier $[[1,1]]$. The seed is irrelevant in evaluation. At $p=0$,
training is also exactly the identity. There is no requirement to drop an
exact fraction of a finite batch; independent draws can drop every entry
or keep every entry.

## Why it works

For a single entry $x$, the survival indicator $B$ equals one with
probability $q$. Consequently $\mathbb{E}[xB/q]=x$. This expectation explains
the training scale factor and the unscaled evaluation rule. With the mask
held fixed, $y=xM$ has derivative $\partial y/\partial x=M$. Multiplying
the upstream signal by the same multiplier follows directly from the chain rule.

The original dropout paper motivates stochastic omission as a regularizer
that discourages brittle co-adaptation [paper: Srivastava et al. 2014,
Dropout: A Simple Way to Prevent Neural Networks from Overfitting,
Journal of Machine Learning Research 15, pp.1929–1958]. Its scaling
presentation and inverted scaling use different placement conventions;
this exercise explicitly chooses training-time scaling.

## Signal vs. noise

Know the mode switch, expectation argument, and cached-mask derivative.
These are the interview essentials. The seed is a reproducibility tool:
in a real training loop, advancing a generator produces fresh masks across
steps. Reconstructing the same seeded generator each step would reuse a
pattern. Our interface deliberately regenerates it so a single operation
and its finite-difference check are deterministic. Channel-wise dropout,
stochastic depth, and tuning schedules can wait.

## Where it breaks

Sampling a new mask during backward differentiates a different realized
function. Scaling both train and evaluation changes expected activations.
Confusing drop probability with keep probability reverses which units survive.
A global random seed creates side effects in unrelated initialization or
sampling code; use a local generator instead.

At $p=1$, division by the keep probability is undefined. This problem caps
$p$ at $0.95$ to keep the multiplier at most twenty. Nonfinite inputs,
empty matrices, and shape mismatches are rejected. For a finite-difference
check, reuse the seed for both positive and negative perturbations. Changing
the mask between evaluations measures random variation rather than a derivative.

## Interview angle

Explain train and evaluation behavior first, then derive the expected-value
scaling and backward rule. A likely follow-up asks whether the evaluation
network exactly equals an ensemble average; nonlinear downstream layers mean
matching activation expectations does not establish that equality. Another
asks why dropout can hurt small networks; removing capacity and injecting
noise can make an already underfit model worse.

## Connections

The operation is a diagonal linear map once its randomness is fixed, making
it a compact application of vector-Jacobian products. It can be inserted
between hidden activations and the next dense layer in an MLP. Later framework
work adds persistent train/eval modes and generator state, while retaining
the same scaling and cached-mask logic.
