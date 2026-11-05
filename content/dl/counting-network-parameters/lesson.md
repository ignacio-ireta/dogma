---
title: Counting network parameters
minutes: 12
min_viable: Read The system and Worked example, then implement p01.
---

## The system

Count stored scalar slots, layer by layer. Let widths $[d_0,d_1,\ldots,d_L]$ describe a chain with $L$ dense layers. Width $d_0$ is the input dimension and $d_\ell$ is layer $\ell$'s output width. Each weight matrix has shape $(d_{\ell-1},d_\ell)$ and each optional bias vector has shape $(d_\ell,)$. No parameters are shared in this exercise.

| Stored object | Count |
|---|---|
| Layer $\ell$ weights | $d_{\ell-1}d_\ell$ |
| Layer $\ell$ bias | $d_\ell$, if enabled |
| Network total | sum of all layer counts |

First action: write the matrix and vector shapes for a two-layer chain. Activations, examples and loss values are not trainable parameter slots. This topic is careful bookkeeping around an already familiar dense layer.

## Worked example

For widths $[3,4,2]$, the first layer has $3\cdot4=12$ weights and $4$ biases, giving $16$. The second layer has $4\cdot2=8$ weights and $2$ biases, giving $10$. The total is $16+10=26$. Without biases the per-layer counts are $[12,8]$, totaling $20$.

A batch of one example and a batch of hundreds use these same 26 parameters. Their activation storage differs. If all parameters were stored as four-byte floating point values, their raw storage would be $26\cdot4=104$ bytes. That arithmetic excludes gradients, optimizer state, metadata and allocator overhead.

## Why it works

A dense output unit learns one coefficient per input feature and optionally one additive constant. With $b$ output units and $a$ input features, there are $b$ groups of $a+1$ learned values when bias is enabled: $b(a+1)=ab+b$. Stacking distinct layers adds their independent storage counts.

This follows directly from the affine-layer construction in [Deep Learning, chapter 6](https://www.deeplearningbook.org/contents/mlp.html). Nonlinear activations such as fixed ReLU do not add parameters. A learnable activation slope would add parameters and must be counted explicitly; it is outside this dense-only calculator.

## Signal vs. noise

The signal is a list of parameter shapes and whether the same storage is reused. Count the output layer, preserve layer order, and separate the weight term from the bias term. Returning per-layer counts makes a total easy to audit and catches omissions that a single final integer hides.

Do not build matrices merely to ask their size. Python integer arithmetic handles very large symbolic width budgets without fixed-width overflow. P01 deliberately returns Python integers so a model-budget estimate cannot silently wrap before the network could even be allocated.

## Where it breaks

The formula changes when parameters are shared. Calling the same layer twice doubles some computation and activations, but does not duplicate its learned storage. Embedding tables, convolutions, normalization and learnable activation parameters also require their own shape rules. State which families are included before giving a total.

The contract accepts only a list or tuple with at least two positive Python integer widths. It rejects boolean and float widths even though Python can treat them numerically. A zero-width layer is not part of this calculator's model definition. Batch size and sequence length are intentionally absent from the interface.

## Interview angle

Lead with scope: a chain of distinct dense layers, with bias enabled or disabled consistently. Draw each parameter shape and sum its entries. If asked whether this predicts runtime memory, explain that it estimates only raw parameter storage. Gradients and optimizer moments can add storage, while activations depend on the workload. If asked about tied weights, count the shared object once and explain the reuse.

## Connections

Dense forward gives the storage shapes; initialization fills those slots and backprop assigns each a gradient. Training a small MLP will turn the symbolic budget into actual arrays. The [local representation-learning note](../../../sources/library/AI-ML/10-Backpropagation/LeCun%20Bengio%20Hinton%20Deep%20Learning%202015.md) provides context for learned layers. Record a per-layer budget as the transition note; it gives the next implementation a concrete shape checklist.
