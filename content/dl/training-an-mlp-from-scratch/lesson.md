---
title: "Training an MLP from scratch"
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 using Hint 1 if needed."
---

## The system

A training step is a transaction: compute predictions, compute one scalar
loss, derive every gradient from the same parameters, then update all
parameters together. The exercise trains a small regression network using
full-batch stochastic gradient descent's update rule, without batch sampling.
Let $X\in\mathbb{R}^{n\times d}$ contain examples and
$y\in\mathbb{R}^{n\times1}$ contain scalar targets. With hidden width $h$,
parameters have shapes $W_1:d\times h$, $b_1:h$, $W_2:h\times1$, and $b_2:1$.

$$H=\tanh(XW_1+b_1),\quad \hat y=HW_2+b_2,\quad
L=\frac{1}{2n}\sum_i(\hat y_i-y_i)^2.$$

Here $H\in\mathbb{R}^{n\times h}$ is the hidden activation, $\hat y$ the
prediction, and $L$ the scalar half-mean-squared loss. Initialize weights
from successive `default_rng(seed).standard_normal` draws, divided by the
square root of their respective input widths; initialize biases to zero.
Record the initial loss, then one post-update loss per step. Zero steps
returns a reproducible initial checkpoint. First action: name the four
parameter shapes.

## Worked example

Consider $X=[[0]]$, $y=[[2]]$, and one hidden unit. Regardless of the sampled
weights, zero biases make the initial hidden activation and prediction zero.
The initial loss is $(0-2)^2/2=2$. Let the sampled output weight be $c$.
The initial output upstream gradient is $-2$. Thus the output bias gradient
is $-2$, the hidden bias gradient is $-2c$, and both weight gradients are zero.

For learning rate $\eta=0.1$, the simultaneous update sets $b_2=0.2$ and
$b_1=0.2c$, leaving both weights unchanged. The new prediction is
$c\tanh(0.2c)+0.2$. Compute and record its loss after the update. This example
makes a common bug visible: zero input blocks the input weight gradient,
but does not block the hidden bias gradient.

## Why it works

Set $E=(\hat y-y)/n\in\mathbb{R}^{n\times1}$. The output parameter gradients
are $H^TE$ and the batch sum of $E$. The hidden preactivation gradient is
$D=(EW_2^T)\odot(1-H^2)$, where $D\in\mathbb{R}^{n\times h}$ and $\odot$ is entrywise multiplication.
The input-layer gradients are $X^TD$ and the batch sum of $D$.

Each update is $\theta\leftarrow\theta-\eta\nabla_\theta L$, where $\theta$
stands for one parameter array. Compute $D$ using the old $W_2$; updating
that matrix early changes the derivative of the objective you just measured.
Backpropagation and gradient descent are distinct stages, as developed in
[book: Goodfellow, Bengio and Courville, Deep Learning, §§6.1, 6.5, 8.1].

## Signal vs. noise

The essential debugging order is shapes, loss scaling, gradients, update
ordering, then learning rate. Reproduce one step before judging a long
learning curve. Our initialization is a fully specified fan-in normal rule;
it is not a request to choose between all Xavier and He variants. Mini-batch
shuffling, momentum, schedules, and model classes are outside this session.
The checkpoint here contains final parameter arrays and a loss history,
without disk I/O or optimizer state.

## Where it breaks

Symmetric zero weights can prevent distinct hidden units from learning
different functions. Very large preactivations saturate tanh and shrink its
local derivative. A large learning rate can increase loss or diverge; loss
is not required to decrease on every dataset. Half-mean-squared loss divides
by examples once. Dividing again during dense backprop slows learning by
another factor of $n$. If a loss checkpoint becomes nonfinite, the bounded
implementation raises ValueError instead of returning a corrupted checkpoint.

Targets must be columns. A target vector may broadcast into a square error
matrix, producing a plausible scalar for the wrong objective. Reject that
shape. Finite differences perturb one initial parameter at a time and compare
the numerical loss derivative with the observed one-step parameter change
divided by the learning rate. This checks the whole gradient chain independently.

## Interview angle

State architecture and loss, trace forward and backward shapes, then describe
simultaneous SGD and reproducible initialization. A likely follow-up asks how
to debug a flat loss; inspect saturation, gradient magnitudes, and one-step
finite differences. Another asks how to resume training; include parameters,
optimizer state if any, and random state when sampling is involved.

## Connections

Dense forward maps, tanh derivatives, initialization, and affine backward
rules now form one complete training loop. The scalar autograd exercise can
automate the calculus, while this implementation keeps array-level formulas
visible. Dropout adds a cached stochastic transformation to this system.
Later framework training loops preserve these same stages while managing
parameters and gradients automatically.
