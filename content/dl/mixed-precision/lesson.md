---
title: "Mixed precision"
minutes: 12
min_viable: "Read The system and the worked example, then do p01 with the update table beside you."
---

## The system

A mixed-precision training step keeps three things apart:

1. **Master weights** in fp32, so small updates are not lost to rounding.
2. **Forward and backward** on fp16 or bf16 copies, under autocast.
3. **Loss scaling**, needed for fp16 and usually not for bf16. The loss is multiplied by a scale $S$ before the backward pass, and the gradients are divided by $S$ before the step.

Two properties of a format differ. **Range** comes from the exponent bits and decides overflow and underflow.
**Precision** comes from the mantissa bits and decides how finely values are rounded.

| Format | Exponent bits (range) | Mantissa bits (precision) | Largest finite | Smallest subnormal | Loss scaling |
|---|---|---|---|---|---|
| fp16 | 5 | 10 (about 3 digits) | 65504 | $2^{-24}\approx6\times10^{-8}$ | needed |
| bf16 | 8 | 7 (about 2.4 digits) | $3.4\times10^{38}$ | $2^{-133}\approx9\times10^{-41}$ | usually not |

The dynamic rule for $S$ runs once per step, with $n$ the growth interval:

| Step outcome | Optimizer step | Scale $S$ | Good-step count |
|---|---|---|---|
| Some gradient is $\pm\infty$ or NaN | skipped | $S \leftarrow S\cdot\text{backoff}$ (0.5) | reset to 0 |
| All gradients finite, count $<n$ | applied | unchanged | count $+1$ |
| All gradients finite, count reaches $n$ | applied | $S \leftarrow S\cdot\text{growth}$ (2) | reset to 0 |

PyTorch's defaults are $S_0=2^{16}$, growth 2, backoff 0.5, and $n=2000$.

## Worked example

Let one true gradient be $g=10^{-8}$. In fp16 it is below half the smallest subnormal ($2^{-25}\approx3\times10^{-8}$),
so it rounds to $0$. This is a range failure.

Scale by $S=2^{10}=1024$. The result, $1.024\times10^{-5}$, is in range but below the normal minimum
$2^{-14}\approx6.1\times10^{-5}$. It is subnormal, so it keeps only 8 significant bits: it rounds to
$172\cdot2^{-24}=1.0252\times10^{-5}$. Dividing by $S$ gives $1.0012\times10^{-8}$, an error of about $0.1\%$.
That error is a precision cost. The scale itself is exact, because $2^{10}$ is a power of two.

Overflow works the other way. A gradient of $0.5$ scaled by $2^{17}$ becomes $65536$, above the fp16 maximum.
It becomes $\infty$, so the step is skipped and $S$ becomes $2^{16}$.

## Why it works

Scaling by a power of two adds no rounding error in the normal range. Choosing $S$ is the trade: large enough to lift
small gradients out of underflow, small enough that the largest gradient stays under $65504$. Loss scaling fixes
underflow only. Overflow is handled by the skip rule.

Dynamic scaling tracks a safe $S$, close to the largest one, without knowing the gradients in advance. An overflow is the signal to back
off, and the skipped step matters because an overflowing gradient must not reach the weights. The ST-MoE report is cited
for fp32 weights in the update and bf16 matrix multiplies [corpus: library-papers/2202.08906.pdf p.8], and the Sparse
Transformer report for dynamic loss scaling against fp16 underflow [corpus: library-papers/1904.10509.pdf p.7].

Master weights are needed even with bf16. The spacing of bf16 near $1$ is $2^{-7}\approx 0.0078$, so an update of
$10^{-4}$ to a weight near $1$ would be lost in a bf16 copy.

## Signal vs. noise

**Signal:** the three parts, the range-versus-precision split, the dynamic update table, the skip on overflow, and
the fact that the scale is a power of two.

**Noise:** the exact list of operations autocast runs in each precision. It varies by device and version, so check dtypes instead.

## Where it breaks

- **Clipping before unscaling.** Clipping must see true gradients, so unscale first.
- **Cancellation, in any format.** Subtracting nearly equal numbers exposes their stored error. In bf16, $1.01$ is
  stored as $1.0078125$, so $x-1$ is $0.0078125$ instead of $0.01$: an error of about 22%.
- **Accumulation.** Once a term is below half the spacing of a running sum, it rounds away. Adding $10^{-4}$ ten
  thousand times in fp16 stalls at $0.25$, while fp32 reaches about $1.0$. Accumulate in fp32, as the
  modern-AI section of the Library note argues for long dot products [source: library-refresh/01-Mathematics/05-Linear-Algebra/1991 -- Goldberg -- Finite Precision, Rounding Error and Stable Computation -- Study Note.md].
- **A scale that is too high.** Repeated overflows skip steps and the scale oscillates near its safe maximum.
- **A scale that is too low.** Dynamic scaling doubles only once per growth interval, so a low start recovers slowly:
  about 2000 good steps per doubling by default. A fixed scale never recovers.
- **Reductions are not automatic.** With bf16 inputs such as a matmul output, `softmax`, `log_softmax`, `layer_norm`,
  `sum` and `mean` returned bf16 under CPU autocast in one run (torch 2.14.1), and `cross_entropy` returned fp32. With
  fp32 inputs, all of them returned fp32. Check the dtypes, and cast with `.float()` where precision matters.

## Interview angle

A calm 60-second answer: fp16 training keeps fp32 master weights and runs forward and backward in fp16 under autocast.
The loss is multiplied by a scale so small gradients survive, and the gradients are unscaled before the update. If any
gradient is infinite or NaN, the step is skipped and the scale halves. After a run of good steps the scale doubles.
bf16 has the fp32 exponent, so it usually runs without scaling, but it keeps fp32 master weights because its mantissa
is short. Full versions are in `interview.md`.

## Connections

This topic needs [one training step](../one-training-step/) and the [floating-point formats](../../inf/floating-point-formats/)
topic, which gives the ranges used here. Goldberg's survey covers the rounding and cancellation points, per the Library
note. Loss scaling and master weights follow Micikevicius et al. 2017 (arXiv:1710.03740), cited from memory; it is not in the corpus. Next: gradient accumulation
and gradient checkpointing.
