---
title: "Backprop through a dense layer"
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 using Hint 1 if needed."
---

## The system

A dense layer is one affine map with three routes for its backward signal.
Let $X\in\mathbb{R}^{n\times d}$ contain a batch of $n$ rows and $d$ features,
$W\in\mathbb{R}^{d\times k}$ contain weights, and $b\in\mathbb{R}^{k}$ contain
biases. The output is $Y=XW+b\in\mathbb{R}^{n\times k}$, with the same bias
broadcast into every row. The caller supplies an upstream signal
$G\in\mathbb{R}^{n\times k}$, the derivative of a scalar objective with respect
to these output entries.

| Destination | Backward rule | Shape |
|---|---|---|
| Input | $dX=GW^T$ | $n\times d$ |
| Weight | $dW=X^TG$ | $d\times k$ |
| Bias | $db=\sum_i G_{i,:}$ | $k$ |

In this exercise, `G` is deliberately unreduced. `sum` applies this table;
`mean` divides **all three** results by $n$. An upstream gradient from an
already averaged loss must use `sum` here. This convention prevents averaging
twice. First action: write the three output shapes before writing any code.

## Worked example

Take one row $X=[2,3]$, weights $W=[4,5]^T$, and bias $b=[1]$.
The forward result is $Y=[24]$. Suppose the scalar objective sends back
$G=[2]$. Each weight influences the output in proportion to its input:
$dW=[4,6]^T$. Each input influences the output through its weight:
$dX=[8,10]$. The bias is added once, so $db=[2]$.

For a reduction example, use two input rows $[1]$ and $[3]$, weight $[2]$,
and upstream rows $[1]$ and $[1]$. The summed results are $dX=[2,2]^T$,
$dW=[4]$, and $db=[2]$. Mean reduction gives $[1,1]^T$, $[2]$, and $[1]$.
The batch axis disappears only from shared parameters; it remains in $dX$.

## Why it works

Write one entry as $Y_{ic}=\sum_j X_{ij}W_{jc}+b_c$. Here $i$ indexes rows,
$j$ features, and $c$ outputs. For a weight entry, the chain rule gives
$\partial L/\partial W_{jc}=\sum_i G_{ic}X_{ij}$, where $L$ is the scalar
objective. This is exactly entry $(j,c)$ of $X^TG$. Summing over examples is
necessary because each example uses the same weight. The bias derivative
similarly sums every place where broadcasting reused that bias.

The computational graph approach in [book: Goodfellow, Bengio and Courville,
Deep Learning, §6.5] explains this as a vector-Jacobian product. We propagate
one objective's sensitivity without materializing an enormous Jacobian.

## Signal vs. noise

The essential skills are tracking dimensions, reversing the map, and naming
where the objective was normalized. Matrix transposes are bookkeeping; the
shared-parameter sum is the conceptual step. Ignore optimizer states and
higher-order derivatives during this session. A useful diagnostic is to
compute one weight derivative by a scalar loop and compare its position with
the matrix result. A numerically plausible answer with the wrong orientation
is still a shape error.

## Where it breaks

Broadcasting can silently turn a column bias gradient into a matrix, so return
exactly shape $(k,)$. A batch of one can conceal mistaken averaging; check a
larger batch too. Integer input must become floating point before division.
Nonfinite values and empty dimensions violate this bounded problem's contract.
The reference rejects them instead of returning meaningless gradients.

Finite differences provide an independent check: perturb each entry of $X$,
$W$, or $b$ and measure the change in $\sum_{i,c}Y_{ic}G_{ic}$. Keep $G$ fixed.
Divide this scalar objective by $n$ for the mean case. Do not compare against
a second implementation of the same backward equations alone.

## Interview angle

Start with the three shapes and state the loss reduction. Then explain that
input gradients travel through weights, while parameter gradients collect
contributions from every example. A likely follow-up asks why bias sums;
broadcasting reused one parameter. Another asks whether to divide by batch
size; inspect the upstream loss convention first.

## Connections

This extends dense forward propagation and scalar chain rules into batched
linear algebra. It supplies the two affine backward steps inside the MLP
training problem. Scalar reverse-mode engines automate the same routing
principle, while framework autograd adds tensor broadcasting and saved-state
management. The historical backpropagation perspective is developed in
[book: Goodfellow, Bengio and Courville, Deep Learning, §6.5].
