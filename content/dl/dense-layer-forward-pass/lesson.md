---
title: Dense layer forward pass
minutes: 12
min_viable: Read The system and Worked example, then implement p01.
---

## The system

A dense layer is an affine map applied independently to each example. Let $X\in\mathbb{R}^{n\times d_{in}}$ hold a batch with $n$ rows, $W\in\mathbb{R}^{d_{in}\times d_{out}}$ hold weights, and $b\in\mathbb{R}^{d_{out}}$ hold biases. The output is $Y=XW+b\in\mathbb{R}^{n\times d_{out}}$. The bias broadcasts across rows; it does not gain a batch dimension as a parameter.

| Object | Rows mean | Columns mean |
|---|---|---|
| $X$ | examples | input features |
| $W$ | input features | output units |
| $Y$ | examples | output units |

Start p01 by writing these three shapes. Check the contracted dimensions before multiplying. Check bias rank separately: a row-shaped bias may broadcast correctly but still violates this exercise's one-dimensional interface. This is interface bookkeeping, not a new mathematical operation.

## Worked example

Take one example $X=[1,2]$, weights $W=[[1,0,2],[0,1,3]]$, and bias $b=[0.5,0,-1]$. There are two input features and three outputs. The dot products are $1\cdot1+2\cdot0=1$, $1\cdot0+2\cdot1=2$, and $1\cdot2+2\cdot3=8$. Add the corresponding bias once: $Y=[1.5,2,7]$.

Add a second row $[3,4]$. Its output is $[3.5,4,17]$. The first row's output stays unchanged. Neither example changes the weights or the bias during this forward pass. Predict that independence before coding the matrix operation.

## Why it works

For output entry $(i,j)$, the rule is $Y_{ij}=\sum_{k=1}^{d_{in}}X_{ik}W_{kj}+b_j$. Matrix multiplication organizes these dot products without Python loops. It does not change their meaning. The same column of weights describes output unit $j$ for every example. The affine-layer construction is developed in [Deep Learning, chapter 6](https://www.deeplearningbook.org/contents/mlp.html).

Two affine layers without a nonlinear activation still collapse to one affine map: $(XW_1+b_1)W_2+b_2=X(W_1W_2)+(b_1W_2+b_2)$. Depth alone therefore does not create the nonlinear features a multilayer network needs.

## Signal vs. noise

The signal is axis meaning, shared parameters and exact output shape. A vectorized implementation is useful because the numerical library handles the repeated contractions. Do not replace shape reasoning with trial transposes until a program runs. A square weight matrix can hide an orientation mistake because either orientation multiplies successfully.

For this session, ignore GPU kernels and framework classes. Convert arithmetic to float64, preserve the batch dimension for one row, and return an array rather than a list. Empty dimensions are legitimate sums and shapes, not special learned behavior.

## Where it breaks

A one-dimensional example loses its batch axis. An incorrectly oriented weight can produce a feature mismatch, or silently compute a different map when square. A column-shaped bias can accidentally add along examples when dimensions coincide. Inputs remain read-only: in-place bias addition to an input view can corrupt upstream activations.

This exercise validates rank and compatible dimensions. It is not a general numeric-input sanitizer. Floating point arithmetic can overflow for extreme products; choosing finite inputs does not guarantee a representable output. The tests use finite representable arithmetic and do not ask for arbitrary-precision matrix products.

## Interview angle

Lead with the shapes, then say which dimension is summed and which bias is shared. Demonstrate one output entry before writing `X @ W + b`. If asked about cost, dense arithmetic scales with the product of batch size, input width and output width. If asked about another library's weight layout, inspect its documented convention and transpose at the boundary. Never infer orientation from the word “linear” alone.

## Connections

Broadcasting is the prerequisite; dense backprop reuses the same axis meanings in reverse. Parameter counting counts the entries in $W$ and $b$, independently of batch size. Initialization chooses their starting values. The [local backpropagation note](../../../sources/library/AI-ML/10-Backpropagation/Rumelhart%20Hinton%20Williams%20Backpropagation%201986.md) connects this forward calculation to hidden-layer credit assignment. After p01, write the four object shapes from memory as the transition note.
