---
title: "Floating-point formats"
minutes: 12
min_viable: "Read The system and the format table, then do p01 with the table beside you."
---

## The system

A floating-point number stores three fields: a sign bit $s$, an exponent field $E$ with $e$ bits, and a mantissa field $M$ with $m$ bits. A format is the triple $(e, m, b)$, where $b$ is the exponent bias. The total width is $1+e+m$ bits.

The value depends on $E$:

| Case | Condition | Value |
|---|---|---|
| Normal | $1\le E\le 2^e-2$ | $(-1)^s\,(1+M/2^m)\,2^{E-b}$ |
| Subnormal | $E=0$ | $(-1)^s\,(M/2^m)\,2^{1-b}$ |
| Infinity | $E=2^e-1$, $M=0$ | $\pm\infty$ |
| NaN | $E=2^e-1$, $M\neq 0$ | not a number |

The leading $1$ of a normal number is implicit, so it costs no bits. Subnormals drop it and share the smallest normal exponent. They fill the gap between zero and the smallest normal.

| Format | $(e, m, b)$ | Largest finite | Smallest normal | Smallest subnormal | Machine epsilon $2^{-m}$ |
|---|---|---|---|---|---|
| fp32 | $(8, 23, 127)$ | $\approx 3.4\times10^{38}$ | $2^{-126}\approx1.2\times10^{-38}$ | $2^{-149}\approx 1.4\times10^{-45}$ | $2^{-23}\approx 1.2\times10^{-7}$ |
| fp16 | $(5, 10, 15)$ | $65504$ | $2^{-14}\approx 6.1\times10^{-5}$ | $2^{-24}\approx 6.0\times10^{-8}$ | $2^{-10}\approx 9.8\times10^{-4}$ |
| bf16 | $(8, 7, 127)$ | $\approx 3.4\times10^{38}$ | $2^{-126}$ | $2^{-133}\approx 9.2\times10^{-41}$ | $2^{-7}=0.0078125$ |
| E5M2 (fp8) | $(5, 2, 15)$ | $57344$ | $2^{-14}$ | $2^{-16}\approx 1.5\times10^{-5}$ | $2^{-2}=0.25$ |
| E4M3 (fp8) | $(4, 3, 7)$ | $448$ | $2^{-6}\approx 0.016$ | $2^{-9}\approx 0.002$ | $2^{-3}=0.125$ |

E5M2 follows the IEEE rules above. E4M3 does not. It has no infinity. The two patterns $E=15$, $M=111$ (one per sign bit) are NaN. The other patterns with $E=15$ are normal. That is why its largest value is $1.75\cdot2^{8}=448$.

## Worked example

Decode fp16 pattern `0 01111 0000000000`. Here $s=0$, $E=15$, $M=0$, $b=15$. It is normal, so the value is $(1+0)\,2^{0}=1$.

Decode `0 00000 0000000001`. Here $E=0$ and $M=1$. It is subnormal, so the value is $(1/1024)\,2^{-14}=2^{-24}\approx 6.0\times10^{-8}$.

Round $1+2^{-8}$ to bf16. Its neighbours are $1$ and $1+2^{-7}$, and the spacing is $2^{-7}$. The input is exactly halfway. Ties go to the even mantissa: $1$ has $M=0$ (even) and $1+2^{-7}$ has $M=1$ (odd). The result is $1$.

Round $1+3\cdot2^{-8}$ to bf16. It lies halfway between $1+2^{-7}$ ($M=1$, odd) and $1+2^{-6}$ ($M=2$, even). The result is $1+2^{-6}=1.015625$.

## Why it works

**Rounding rule.** Round to nearest, ties to even. Inside one binade $[2^k,2^{k+1})$ the representable numbers are evenly spaced at $2^{k-m}$. A real $x$ maps to the nearest one. The relative error is at most $u=2^{-(m+1)}$, the unit roundoff, which is half the machine epsilon.

**Range versus precision.** The exponent field sets the range and the mantissa field sets the precision. fp16 has $e=5$, so it overflows above $65504$ and underflows below about $6\times10^{-8}$. bf16 has $e=8$, the same as fp32, so its range matches fp32. It keeps only $m=7$ mantissa bits, so its relative error is about $0.4\%$ per rounding, against about $6\times10^{-8}$ for fp32. A bf16 value is the top 16 bits of an fp32 value after rounding, which is why conversion is cheap.

**Where bf16 is used.** The ST-MoE report trains with fp32 weights, bf16 matrix multiplies and bf16 activations. It notes that bf16 roundoff on large numbers can be up to $65{,}536$ times the fp32 roundoff, so the range is fine and the precision is what breaks [corpus: library-papers/2202.08906.pdf p.9].

**Why fp16 needs more care.** Any value below half the smallest subnormal, about $3\times10^{-8}$, rounds to zero in fp16. Small gradients are exactly that kind of value. A loss scale moves them into range, as the mixed-precision topic shows.

## Signal vs. noise

**Signal:** the three numbers per format (largest finite, smallest subnormal, machine epsilon), the rounding rule, and the trade that bf16 makes between mantissa and range.

**Noise:** NaN payloads and hardware bit tricks. Know that $-0$ exists, then move on.

## Where it breaks

- **Overflow to infinity.** A fp16 activation above $65504$ becomes $\pm\infty$, and the next operation spreads it. Check the magnitude of attention logits and residual streams.
- **Absorption.** Adding a small number to a large one can change nothing in low precision. In fp16, $2048+1$ returns $2048$, because the spacing at $2048$ is $2$. Long sums need an fp32 accumulator.
- **Exponential sensitivity.** The ST-MoE report gives a softmax example. Logits $128$ and $128.5$ become equal after bf16 rounding, and the softmax output changes by about $36\%$ [corpus: library-papers/2202.08906.pdf p.9]. Keep the softmax and its logit reductions in fp32.
- **fp8 has very few mantissa bits.** E4M3 has $m=3$, so its epsilon is $0.125$. The DeepSeek-V4 report keeps FP32 master weights, quantizes them to FP4, and dequantizes to FP8 with scale factors on small tiles, which keeps the range inside E4M3 [corpus: library-papers/deepseek-v4-2026.pdf p.33].

## Interview angle

A calm 60-second answer: the format is sign, exponent and mantissa. The exponent decides overflow and underflow. The mantissa decides rounding error. fp16 has a small exponent, so it needs loss scaling. bf16 has the fp32 exponent, so it does not, but each rounding costs about $0.4\%$. Full versions are in `interview.md`.

## Connections

This topic has no prerequisites. It leads to the mixed-precision topic, where loss scaling and autocast use these rules, and to int8 quantization, where the same rounding error is controlled by a scale and a zero point.
