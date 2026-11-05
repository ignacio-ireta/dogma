---
title: "Arithmetic intensity and the roofline"
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 with the three counts (FLOPs, bytes, ridge) written on paper first."
---

## The system

A device has two limits: at most $P$ FLOP/s of compute, and at most $\beta$ bytes/s of memory bandwidth. An
operation that performs $F$ FLOPs and moves $Q$ bytes cannot finish faster than either limit allows:

$$
t \ge \max\left(\frac{F}{P},\ \frac{Q}{\beta}\right)
$$

| Quantity | Formula | Unit |
|---|---|---|
| Arithmetic intensity | $I = F / Q$ | FLOP per byte |
| Ridge point | $R = P / \beta$ | FLOP per byte |
| Attainable rate | $\min(P,\ I\beta)$ | FLOP/s |

Rules:

1. $I < R$: **memory-bound**. The bandwidth roof $I\beta$ is the lower one.
2. $I \ge R$: **compute-bound**. The compute roof $P$ is the lower one. At exactly $I = R$ we report compute-bound.
   The tie rule is a convention, and the problem states it.

**Matrix product.** Let $A$ be $m \times k$, $B$ be $k \times n$ and $C = AB$ be $m \times n$, with $b$ bytes per element.

- FLOPs: $F = 2mnk$. Each of the $mn$ outputs is a dot product of length $k$: $k$ multiplies and $k$ adds.
- Bytes: $Q = (mk + kn + mn)\,b$. Each element of $A$ and $B$ is read once and each element of $C$ is written once.
  No cache reuse is credited, so $Q$ is an ideal count.

## Worked example

Device: $P = 312$ TFLOP/s $= 3.12 \times 10^{14}$ FLOP/s, $\beta = 2$ TB/s $= 2 \times 10^{12}$ B/s, so $R = 156$
FLOP/byte. Use fp16, $b = 2$.

**Batch-1 decode.** One token times a weight matrix: $m = 1$, $k = n = 4096$.

- $F = 2 \cdot 1 \cdot 4096 \cdot 4096 = 33{,}554{,}432$ FLOP.
- $Q = (4096 + 4096^2 + 4096) \cdot 2 = 33{,}570{,}816$ bytes.
- $I = 0.9995$ FLOP/byte, below $R = 156$: **memory-bound**.
- Attainable: $0.9995 \cdot 2 \times 10^{12} \approx 2.0 \times 10^{12}$ FLOP/s, which is 0.64 % of $P$.
- Time floor: $\max(0.108\ \mu s,\ 16.79\ \mu s) = 16.79\ \mu s$. The memory term binds.

**Square product.** $m = n = k = 4096$: $F = 137{,}438{,}953{,}472$ FLOP, $Q = 100{,}663{,}296$ bytes, $I = 1365.3$.
That is above $R$, so the product is **compute-bound**. Its time floor is $F/P = 0.4405$ ms, against $Q/\beta = 0.050$ ms.

**The tie.** A device with $P = 1000$ and $\beta = 10$ has $R = 100$. Take $m = n = k = 150$ and $b = 1$: $F = 6{,}750{,}000$,
$Q = 67{,}500$, and $I = 100 = R$ exactly. Both time floors equal 6750 s, and the tie rule reports **compute-bound**.

## Why it works

Each limit gives its own time, and the slower one dominates, so the floor is the maximum. The attainable rate restates
the bound: a memory-bound kernel consumes at most $\beta$ bytes per second, which is $I\beta$ FLOP/s.
The roofline is an optimistic bound. Real kernels reach a fraction of it.

**Why batch-1 decode is memory-bound.** With $m$ much smaller than $k$ and $n$, the weights dominate:
$Q \approx kn\,b$, so $I \approx 2m/b$. In fp16 with $m = 1$, $I \approx 1$, about 156 times below the ridge. Each
weight is loaded once and used for one multiply-add. Batching raises $m$, and $I$ with it, until $I$ reaches
$R$ near $m = R\,b/2 = 156$ sequences.

Literature. The compute-to-memory ratio estimates an operation's efficiency, and a model is memory-bound when loading
tensors greatly exceeds computing on them [corpus: library-papers/2202.08906.pdf p.14]. Compute-bound and memory-bound
follow which work sets the time [corpus: library-papers/2205.14135.pdf p.3]. Williams, Waterman and Patterson (2009)
introduced the roofline; the corpus cites it only in a reference list [corpus: library-papers/2312.00752.pdf p.22].

## Signal vs. noise

- **Signal:** the ridge $R = P/\beta$ and where $I$ sits against it. Those two numbers name the bound.
- **Signal:** the factor 2 in $F = 2mnk$ and the three terms in $Q$. Missing either shifts $I$ by a constant factor.
- **Noise for now:** the $-1$ in "$2k - 1$ FLOPs per output", tiling, and cache hit rates.

## Where it breaks

- **MACs instead of FLOPs.** One multiply-add is 2 FLOPs. Counting $mnk$ halves $I$.
- **Forgetting the output.** $C$ is written, so $mn$ elements belong in $Q$.
- **Measured time as intensity.** $I$ is a property of the operation, not of a run.
- **Mixed peaks.** Use the peak for the dtype you run; a tensor-core fp16 peak differs from fp32.
- **Units.** TB and TiB differ by about 10 percent. Write the unit beside each number.

## Interview angle

**60-second answer.** Intensity is FLOPs divided by bytes moved. The ridge is peak FLOP/s divided by bandwidth. Below
the ridge the operation is memory-bound, and its time is bytes over bandwidth. Above it, compute-bound. Batch-1 decode
sits near 1 FLOP/byte in fp16, far below a ridge in the hundreds, so decode is memory-bound. A long prefill is a large
matmul with high intensity, so it is compute-bound.

**Likely follow-ups.** Why is decode intensity about $2/b$? Each weight is read once and used for two FLOPs, and weight
bytes dominate the traffic. Why does batching help? It reuses each loaded weight across more sequences.

## Connections

This topic needs matrix products (`cml/matrix-multiplication`) and element widths (`inf/floating-point-formats`). It leads to prefill versus decode bottlenecks and to GPU memory sizing, where the same weight bytes set how many devices a model needs.
