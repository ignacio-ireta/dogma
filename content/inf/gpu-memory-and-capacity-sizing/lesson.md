---
title: "GPU memory and capacity sizing"
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 by writing the per-GPU total for T = 1, 2, 4 and 8 before coding."
---

## The system

A model replica served across $T$ GPUs (tensor-parallel degree $T$) must fit its per-GPU share of everything it keeps
in memory. Let $M$ be the memory of one GPU, $r$ the reserve fraction, $A$ the workspace per GPU, $W$ the weight bytes,
$B$ the concurrent sequences, $S$ the tokens stored per sequence, and kv the cache bytes per token per sequence (the
full model, from the KV-cache topic). With $H_{kv}$ KV heads:

$$
\underbrace{\left\lceil \frac{W}{T} \right\rceil}_{\text{weights}}
+ \underbrace{\frac{B\,S\,\text{kv}}{T}}_{\text{KV cache}}
+ \underbrace{A}_{\text{workspace}}
\le \underbrace{\left\lfloor M\,(1-r) \right\rfloor}_{\text{usable}}
$$

Four rules set the answer:

1. **Weights are sharded by $T$.** Each GPU holds $1/T$ of them; the ceiling keeps the share an integer.
2. **The KV cache is sharded by KV heads.** Each KV head lives on one GPU, so the share is $1/T$ of the cache, which is
   exact only when $T$ divides $H_{kv}$.
3. **Degrees are powers of two that divide $H_{kv}$.** With 8 KV heads the candidates are 1, 2, 4 and 8.
4. **The answer is the smallest candidate that fits.** More GPUs shrink both shares, so the first fit is the minimum.
   If none fits, the answer is zero.

## Worked example

A 70B-class model: $L = 80$, $H_{kv} = 8$, $d_h = 128$, fp16, so $W = 140$ GB and kv $= 327{,}680$ bytes per token.
The GPUs have 80 GiB ($85{,}899{,}345{,}920$ bytes), with $A = 4$ GiB of workspace and a 10 percent reserve. The
usable memory is $\lfloor 0.9 \cdot 85{,}899{,}345{,}920 \rfloor = 77{,}309{,}411{,}328$ bytes, which is 72 GiB.

**Load of 32 sequences of 4096 tokens.** The cache holds $32 \cdot 4096 \cdot 327{,}680 = 42.9$ GB.

| $T$ | Weights per GPU | KV per GPU | Total (with 4.3 GB workspace) | Fits in 77.3 GB? |
|---|---|---|---|---|
| 1 | 140.0 GB | 42.9 GB | 187 GB | no |
| 2 | 70.0 GB | 21.5 GB | 95.8 GB | no |
| 4 | 35.0 GB | 10.7 GB | 50.0 GB | **yes** |

The answer is $T = 4$. A single sequence of 4096 tokens needs only $T = 2$: the total is $70.0 + 0.67 + 4.3 = 75.0$ GB.

**Concurrency at $T = 4$.** Spare memory per GPU is $77{,}309{,}411{,}328 - 35{,}000{,}000{,}000 - 4{,}294{,}967{,}296 =
38{,}014{,}444{,}032$ bytes. One sequence of 4096 tokens costs $335{,}544{,}320$ bytes per GPU. The floor of the ratio is
113 sequences.

## Why it works

Tensor parallelism splits each layer's matrices across the group (Shoeybi et al. 2019; not in the corpus). One MoE
study reports model parallelism over the columns of a logical device mesh [corpus: library-papers/2202.08906.pdf p.14].
Each layer then needs communication between GPUs, a cost this model does not count. The KV cache follows the heads: the GPU that
computes a head's attention keeps that head's keys and values. So the KV split is $1/T$ only when every GPU owns a whole
number of KV heads, and that is why $T$ must divide $H_{kv}$. With grouped-query attention, 8 KV heads cap pure tensor
parallelism at 8 GPUs unless heads are replicated, and replication is outside this model.

The reserve and the workspace are per-GPU overhead that does not shrink with $T$. The reserve absorbs allocator
fragmentation and runtime buffers, and the workspace holds activations in flight. That is why adding GPUs eventually
stops helping a small model.

## Signal vs. noise

- **Signal:** weights shard by $T$; KV shards by KV heads.
- **Signal:** the power-of-two and KV-head divisibility rules. They often decide the answer.
- **Noise for now:** the exact allocator, pipeline parallelism, CUDA context size and communication buffers. They are
  real, and they move the answer by a few percent.

## Where it breaks

- **GiB and GB mixed.** 80 GiB is 85.9 GB. Quote units with every number.
- **Sharding the weights but not the cache.** That is an error of a factor $T$ in one of the two large terms.
- **Splitting the cache by query heads.** The split follows KV heads. Using $H_q$ instead is off by $H_q / H_{kv}$, which is wrong under grouped-query attention.
- **A count quoted without its load.** The answer depends on batch and context. State the target you size for.
- **Fit read as speed.** Fitting says the memory works. It says nothing about throughput or the communication cost of
  more GPUs.
- **Ignoring the largest context.** Size for the longest context you allow.

## Interview angle

**60-second answer.** Count each GPU's share: weights divided by the degree, the cache divided by the degree through the
KV heads, and the workspace. Subtract the reserve from the memory first. Try the powers of two that divide the KV heads,
and take the smallest that fits. The leftover over the per-sequence cache gives the concurrency. Name the units and the
load you are sizing for.

**Likely follow-ups.** Why must the degree divide the KV heads? Each KV head must live on one GPU, or the cache cannot
be split evenly. What if the model does not fit even at 8 GPUs? Quantise the weights (fp8 halves the weight bytes), or
quantise the cache, or change the model's head layout.

## Connections

This topic needs the KV-cache bytes per token (`inf/kv-cache-size-arithmetic`) and the parameter memory footprint
(`llm/model-parameter-and-memory-footprint`). It leads to the paged KV cache, which shrinks fragmentation, and to the communication cost of extra GPUs.
