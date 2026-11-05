---
title: "LoRA parameter and memory arithmetic"
minutes: 12
min_viable: "Read The system and the Worked example, then do p01 with Hint 1 open."
---

## The system

A LoRA adapter on a matrix with input width $d_\text{in}$ and output width $d_\text{out}$ holds a down
projection $A \in \mathbb{R}^{r \times d_\text{in}}$ and an up projection $B \in \mathbb{R}^{d_\text{out} \times r}$.
Its trainable count is

$$
N_\text{matrix} = r\,(d_\text{in} + d_\text{out}).
$$

Full fine-tuning of the same matrix trains $d_\text{in} \cdot d_\text{out}$ numbers.

**Which matrices a target name adds.** Each name adds one adapter per block, and nothing else (no embeddings,
no norms, no output head). Model width $d$, MLP width $F$:

| Target | Matrix | $d_\text{in}$ | $d_\text{out}$ | Trainable per block |
|---|---|---|---|---|
| `q`, `k`, `v`, `o` | attention projection | $d$ | $d$ | $2rd$ each |
| `gate`, `up` | MLP input projection | $d$ | $F$ | $r(d+F)$ each |
| `down` | MLP output projection | $F$ | $d$ | $r(F+d)$ |

Multiply the per-block sum by the number of blocks $L$. Grouped-query attention gives `k` and `v` a smaller
$d_\text{out}$, so this table is the full-width case. The fraction is $N_\text{trainable} / N_\text{base}$ when
the base count is known. Otherwise, say which base the fraction is over.

**Memory, in bytes.** The adapter costs its weights and one gradient per weight, each in $b_\text{ad}$ bits, plus two fp32 Adam
moments, 8 bytes per adapter number. Assumptions: the adapter weights are stored in $b_\text{ad}$ bits; the gradients match them; no fp32
master copy of the adapter is kept. Mixed-precision trainers often keep one, which adds 4 bytes per number. With
$N_\text{base}$ base numbers stored at $b_\text{base}$ bits and $N_\text{lora}$ adapter numbers:

$$
\text{bytes} = \left\lceil \frac{N_\text{base}\, b_\text{base}}{8} \right\rceil
+ 2\left\lceil \frac{N_\text{lora}\, b_\text{ad}}{8} \right\rceil + 8\, N_\text{lora}.
$$

Each stored tensor is rounded up to whole bytes on its own. Activations and the framework are left out.

## Worked example

Take $L = 32$, $d = 4096$, $F = 11008$, $r = 16$ and all seven targets. These are the shapes of a 7B-class decoder,
not a specific model.

- Attention per block: $4 \cdot 2 \cdot 16 \cdot 4096 = 524{,}288$.
- MLP per block: gate, up and down are $16 \cdot (4096 + 11008) = 241{,}664$ each, so $724{,}992$.
- Per block: $1{,}249{,}280$. Across 32 blocks: $39{,}976{,}960$, about 40.0 million numbers.

Full updates of the same matrices: $L(4d^2 + 3dF) = 6{,}476{,}005{,}376$. Against that, the adapter is $0.62\%$.
Against a base of $7\cdot10^9$ numbers, it is $0.57\%$.

Memory. The base is $N_\text{base} = 7\cdot10^9$, a round figure for a 7B-class model. The adapted matrices alone
hold $6.48\cdot10^9$ numbers. Embeddings and norms sit inside the round figure. Full fp32 Adam holds weights, gradients and moments, $4 + 4 + 8 = 16$ bytes per parameter, so
$16 \cdot 7\cdot10^9 = 112$ GB. LoRA with a bf16 base: $14$ GB of base, plus
$39{,}976{,}960 \cdot (2 + 2 + 8) = 479{,}723{,}520$ bytes (about 480 MB) for the adapter: bf16 weights (2 bytes),
bf16 gradients (2) and fp32 moments (8), with no fp32 master copy. With a 4-bit base the
base is $3.5$ GB.

## Why it works

The adapter count is linear in $r$ and in the sum of the two widths. Full fine-tuning is quadratic in the width,
so the adapter stays small as the model grows. This is a parameter budget for the update, not a claim that the
base matrix is low rank. Truncated SVD gives the best rank-$r$ approximation of a fixed matrix (Eckart and Young,
1936, [library:AI-ML/02-Linear-Algebra/Eckart-Young Best Low-Rank Approximation 1936.md]), but an adapter does not
approximate the base. The Library bridge note states the same for the update: rank at most $r$, and $r(m+n)$
parameters, per the note [source: library-refresh/01-Mathematics/05-Linear-Algebra/1990-2021 -- Library -- From Latent Semantic Analysis to Embeddings and Low-Rank Adaptation -- Concept Note.md].

Adam keeps two moments per trained number (Kingma and Ba, 2015, [library:AI-ML/03-Optimization/Kingma Ba Adam 2015.md]). So the
optimizer term follows the trainable count, not the base. The frozen base needs storage only: no gradient, no moments.

A 4-bit base is four times smaller than a 16-bit base in storage. Quantization constants are not counted here.

## Signal vs. noise

Signal: the per-matrix formula, the factor $L$, the rounding per tensor, and which term each byte belongs to. Noise:
the allocator, the CUDA context and the dtype names.

## Where it breaks

- **Forgetting the layer factor.** The count is per block. Multiply by $L$ once, not once per target.
- **Counting $d_\text{in} \cdot d_\text{out}$.** That is full fine-tuning. LoRA counts the sum of the two widths.
- **Bits and bytes.** $N \cdot b$ is bits. Divide by eight, then round up per tensor.
- **Believing LoRA cuts weight memory.** It cuts gradients and optimizer states. The base is still stored, unless it is quantized.
- **Ignoring activations.** For long sequences they can outweigh the adapter term. Checkpointing trades them for compute.
- **Forgetting the master copy.** Mixed-precision trainers may add 4 bytes per adapter number on top of the rule above.

## Interview angle

Sixty-second answer: an adapter on a $d_\text{in}$ by $d_\text{out}$ matrix holds $r(d_\text{in} + d_\text{out})$ numbers.
Multiply by the adapted matrices per block and by $L$. The frozen base costs its storage bits. The adapter costs
weights, gradients and 8 bytes of fp32 Adam per number. Likely follow-up: why is the optimizer term the big win? It scales with trainable numbers, not with the model.
Second follow-up: what does the rule leave out? Activations, which depend on batch and sequence length.

## Connections

- Prerequisites: [Model parameter and memory footprint](../model-parameter-and-memory-footprint/) for the base count
  and bytes per parameter, and [LoRA adapters](../lora-adapters/) for the adapter shapes.
- Next: adapter variants and the ways they trade parameters for compute, which build on the same budget.
