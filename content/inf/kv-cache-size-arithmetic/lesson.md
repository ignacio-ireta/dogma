---
title: "KV-cache size arithmetic"
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 with the factors written out on paper first."
---

## The system

While decoding, each token attends to every earlier token. The server keeps one key and one value vector per
token, per layer, per KV head. The cache size is a product of numbers you can read from a model config:

$$
\text{bytes} = \underbrace{2}_{K,\ V} \times L \times H_{kv} \times d_h \times S \times B \times b
$$

| Symbol | Meaning | Typical source in a config |
|---|---|---|
| $L$ | number of layers that keep a cache | `num_layers` |
| $H_{kv}$ | number of KV heads per layer | `num_key_value_heads` |
| $d_h$ | width of one head | `head_dim` |
| $S$ | tokens stored per layer, $S = \min(\text{context}, W)$ | context length; $W$ = sliding window |
| $B$ | sequences held at once (the batch) | scheduler decision |
| $b$ | bytes per element | dtype |

Four rules set the inputs:

1. **Count KV heads, not query heads.** The query-head count and the hidden size do not appear in the
   formula. Grouped-query attention (GQA) sets $H_{kv}$ below the query-head count. Multi-query attention
   (MQA) sets $H_{kv} = 1$.
2. **The factor 2 is K and V.** Both are stored, for every token.
3. **A sliding window caps $S$.** With window $W$, each layer keeps at most $W$ tokens. Without a window,
   $S$ is the full context.
4. **The dtype sets $b$.** fp32 is 4 bytes, fp16 and bf16 are 2, fp8 is 1.

Per token and per sequence the cache costs $2 L H_{kv} d_h b$ bytes; multiply by $S$ and $B$. GiB means
$2^{30}$ bytes; a decimal GB means $10^9$. Keep them apart.

## Worked example

Take a 7B-class multi-head model: $L = 32$, $H_{kv} = 32$, $d_h = 128$, fp16 so $b = 2$.

- Per token: $2 \cdot 32 \cdot 32 \cdot 128 \cdot 2 = 524{,}288$ bytes, which is 512 KiB.
- One sequence of $S = 4096$ tokens: $524{,}288 \cdot 4096 = 2{,}147{,}483{,}648$ bytes, which is exactly 2 GiB.
- Batch $B = 6$: $12$ GiB $= 12{,}884{,}901{,}888$ bytes.
- Weights: $7 \times 10^9$ parameters at 2 bytes is $14 \times 10^9$ bytes, about 13.04 GiB.
- Batch $B = 7$: $15{,}032{,}385{,}536$ bytes, which is more than the weights. The cache overtakes the weights here.

Now change two things. Use 8 KV heads (GQA) and a sliding window of $W = 4096$ on a 32,768-token context.

- Per token: $2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131{,}072$ bytes, a quarter of before.
- Stored tokens per layer: $\min(32768, 4096) = 4096$.
- One sequence: $131{,}072 \cdot 4096 = 536{,}870{,}912$ bytes, which is 512 MiB. Without the window it would be 4 GiB.

fp8 instead of fp16 halves every number above; the 4096-token multi-head sequence becomes 1 GiB.

## Why it works

Attention for token $t$ reads $K_s$ and $V_s$ for every cached position $s < t$ at each layer [corpus: library-papers/1706.03762.pdf p.4]. Storing them costs memory linear in $S$; recomputing them costs compute every step. Each
KV head stores K and V of width $d_h$, so the cache scales with $H_{kv} \cdot d_h$. GQA shares one K and V head across a group of query heads, so it divides the cache by the group size. [corpus: library-papers/deepseek-v4-2026.pdf p.13 uses GQA8 as its baseline]

## Signal vs. noise

- **Signal:** $L$, $H_{kv}$, $S$, $B$ and $b$. These five numbers move the cache by orders of magnitude.
- **Signal:** whether the window applies to every layer. Mixed full and windowed layers need a per-layer sum.
- **Noise:** hidden size, query-head count and vocabulary size. They set weight size and compute, not the cache.

## Where it breaks

- **Query heads instead of KV heads.** Overestimates by $H_q / H_{kv}$, which is 4x for 32 query heads over 8 KV heads.
- **Dropping the factor 2.** Underestimates by half.
- **Window applied as a multiplier.** The window caps $S$; it does not scale the result. Stored tokens are
  $\min(S, W)$, so a window larger than the context changes nothing.
- **Mixing GB and GiB.** 14 GB of weights is about 13.04 GiB. Quote units with every number.
- **Batch as request count.** $B$ counts sequences alive at the same time, not requests served per hour.
- **Ignoring growth.** $S$ is prompt plus generated tokens. Size for the largest context you allow.
- **Quantised caches.** An fp8 cache changes $b$, but per-block scales add bytes back.

## Interview angle

**60-second answer.** The cache equals two times layers times KV heads times head dimension times tokens
times batch times bytes per element. For a 7B-class multi-head fp16 model that is 512 KiB per token and 2 GiB
at 4096 tokens. GQA with 8 KV heads cuts it by four, and a sliding window stops it growing past the window.
The cache overtakes the weights at some batch size, which bounds concurrency.


## Connections

This topic needs the head count and head width from multi-head attention (`dl/multi-head-attention`).
It unlocks GPU memory and capacity sizing, paged KV-cache allocation, quantised KV caches and prefix caching,
where the same arithmetic decides how many blocks fit. Lineage: multi-head attention first, then multi-query
and grouped-query attention, which trade heads for memory.
