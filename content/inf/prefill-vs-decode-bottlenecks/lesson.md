---
title: "Prefill vs decode bottlenecks"
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 by writing the four byte and FLOP counts for one prefill and one decode step."
---

## The system

A dense decoder serves a request in two phases with different shapes of work. Let $P$ be the parameters that take part
in a matmul per token, $b$ the bytes per element (weights and cache share it), $S$ the prompt length, $B$ the batch,
$C$ the tokens already cached, and $\text{kv}$ the cache bytes per token per sequence:

$$
W = P\,b, \qquad \text{kv} = 2\,L\,H_{kv}\,d_h\,b
$$

Here $W$ is the weight bytes read once per pass, and the factor 2 in kv is K and V.

| Phase | Tokens per pass | FLOPs $F$ | Bytes $Q$ | Bound at useful sizes |
|---|---|---|---|---|
| Prefill, $B$ prompts of $S$ tokens | $BS$ at once | $2PBS$ | $W + BS\cdot\text{kv}$ | compute, for long prompts |
| Decode step, $B$ sequences, $C$ cached | $B$ | $2PB$ | $W + B(C+1)\,\text{kv}$ | memory |

Each phase time is the roofline floor $\max(F/P_{\text{peak}},\ Q/\beta)$. The difference: prefill reads the weights once
for $BS$ tokens, while decode reads them once per step for only $B$ tokens. Each sequence also reads its own cache,
which is never shared.

## Worked example

Take a 7B-class model: $P = 7\times10^9$, fp16 so $b = 2$, $W = 14$ GB. Use $L = 32$, $H_{kv} = 32$, $d_h = 128$, so
kv $= 524{,}288$ bytes per token. The device has $3.12 \times 10^{14}$ FLOP/s and $2 \times 10^{12}$ B/s, a ridge of 156
FLOP/byte.

- **Prefill, 512 tokens, one prompt.** $F = 2 \cdot 7\times10^9 \cdot 512 = 7.17$ TFLOP, so $F/P_{\text{peak}} = 23.0$ ms.
  Bytes are $14 \text{ GB} + 512 \cdot 524{,}288 = 14.27$ GB, which takes 7.1 ms. Compute binds: prefill is compute-bound.
- **Prefill, 1 token.** $F/P_{\text{peak}} = 0.045$ ms, while the weight read takes 7.0 ms. Memory binds.
- **Decode, batch 1, $C = 512$.** The step reads 14.27 GB, so the time is 7.13 ms. That is 140 tokens/s.
- **Decode, batch 8, $C = 512$.** The step reads $14 + 8 \cdot 513 \cdot 524{,}288 \approx 16.15$ GB, which takes 8.08 ms.
  That is 991 tokens/s, about 7.1 times the batch-1 rate, for 13 percent more time per step.

Batching is cheap because the 14 GB weight read is shared.

## Why it works

**Prefill is a big matmul.** Stacking $BS$ token rows gives weight-only intensity $I = 2BS/b$. With 512 tokens in fp16
that is $I = 512$ FLOP/byte, above the ridge of 156, so long prompts saturate the compute roof.

**Decode is a matrix-vector product, per sequence.** Its intensity is $I = 2PB/(Pb + B(C+1)\,\text{kv})$. Batching raises
$I$ while the weights dominate, so throughput rises almost linearly with $B$ at first.

**A batch threshold exists only if the cache is small.** As $B \to \infty$, $I$ tends to $2P/((C+1)\,\text{kv})$.
For the multi-head model at $C = 512$ that limit is $1.4 \times 10^{10} / 2.69 \times 10^{8} \approx 52$ FLOP/byte,
which is below the ridge of 156. No batch makes that decode compute-bound. With 8 KV heads the limit is about 208, so a
batch near 623 is compute-bound. Grouped-query attention therefore changes the batch threshold, not just the memory.

Literature. A forward pass costs about $2N$ FLOPs per token, with $N$ the non-embedding parameters
[corpus: library-papers/2001.08361.pdf p.7]. For a state-space (Mamba) model, DRAM reads of the state cache become a
major bottleneck to decoding in memory-bound settings [corpus: library-papers/2604.12374.pdf p.35].

## Signal vs. noise

- **Signal:** what each pass shares. Prefill shares the weights over the prompt; decode shares them over the batch.
- **Signal:** the KV term, which is per sequence and never shared. It sets whether a batch threshold exists.
- **Noise for now:** attention score FLOPs, norms, embedding lookups and launch overhead, all excluded from the model.

## Where it breaks

- **Weights read per sequence.** That gives a decode step $B$ times too long. Read them once per step.
- **Prefill time as TTFT.** TTFT also includes queue wait. The prefill time is only its compute and memory part.
- **Ignoring the cache at long context.** At $C = 8192$ the KV term passes the weights from a batch of four.
- **Batch without a memory check.** The cache must also fit in memory (next topics).
- **Peak as prediction.** The floors assume peak FLOP/s and full bandwidth. Real steps are slower, so use these as
  bounds for comparison, not as predictions.

## Interview angle

**60-second answer.** Prefill processes the whole prompt in one parallel pass, so for long prompts it is compute-bound. Decode
makes one token per sequence per step and streams the weights and each sequence's cache, so it is memory-bound. Batching
amortises the weight read, so throughput rises close to linearly until the cache traffic or memory capacity takes over.
Name the phase before you name the bottleneck.

**Likely follow-ups.** Why is a one-token prefill memory-bound? The weight read dominates the bytes and the FLOPs are
tiny. What raises decode throughput most? A bigger batch, fewer KV bytes (grouped-query attention, a quantised cache)
and fewer weight bytes (quantised weights). Only the bigger batch amortises the weight read; the other two cut bytes per step.

## Connections

This topic needs the roofline (`inf/arithmetic-intensity-and-the-roofline`) and cache bytes per token
(`inf/kv-cache-size-arithmetic`). It leads to GPU memory and capacity sizing, which bounds the batch, and to
continuous batching and chunked prefill, which schedule both phases.
