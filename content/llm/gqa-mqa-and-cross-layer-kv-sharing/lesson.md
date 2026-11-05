---
title: "GQA, MQA and cross-layer KV sharing"
minutes: 12
min_viable: "Read The system and the cache table, then do p01 with the one-token example beside you."
---

## The system

Let $n_h$ be the number of query heads, $n_{kv}$ the number of key/value heads, and $d_h=d_{model}/n_h$ the width of one head. Grouped-query attention needs $n_{kv}$ to divide $n_h$. Let $g=n_h/n_{kv}$ be the number of query heads per group.

Query head $h$ (from $0$ to $n_h-1$) reads key and value head

$$
\text{kv}(h)=\left\lfloor \frac{h}{g} \right\rfloor .
$$

The groups are contiguous. Heads $0..g-1$ share key/value head $0$, the next $g$ heads share head $1$, and so on. Each query head still runs its own softmax over the keys. Only the stored keys and values are shared.

| Variant | $n_{kv}$ | Query head $h$ reads key/value head | Cache values per token per layer |
|---|---|---|---|
| MHA | $n_h$ | $h$ | $2\,n_h\,d_h$ |
| GQA | $1<n_{kv}<n_h$ | $\lfloor h/g\rfloor$ | $2\,n_{kv}\,d_h$ |
| MQA | $1$ | $0$ for every head | $2\,d_h$ |

The cache column counts the keys and values that must be stored for each token in each layer, because decoding reads them again at every step.

## Worked example

Take a model with $n_h=32$ query heads, $d_h=128$, $32$ layers, bf16 storage (2 bytes per value), and a $4096$-token context for one sequence.

- MHA stores $2\cdot 32\cdot 128=8192$ values per token per layer, which is $16$ KiB. Over $32$ layers and $4096$ tokens this is $2$ GiB.
- GQA with $n_{kv}=8$ stores a quarter of that, so $512$ MiB.
- MQA with $n_{kv}=1$ stores $1/32$ of MHA, so $64$ MiB.

The mapping is just as concrete. With $n_h=4$ and $n_{kv}=2$, we have $g=2$. Query heads $0,1$ read key/value head $0$, and heads $2,3$ read head $1$.

Published configurations follow the same pattern. The DeepSeek-V4 report compares its KV cache against a GQA model with 8 KV heads as the baseline [corpus: library-papers/deepseek-v4-2026.pdf p.13]. Nemotron 3 Super uses 32 query heads with 2 KV heads, a group size of 16 [corpus: library-papers/2604.12374.pdf p.8].

## Why it works

Cache size is linear in $n_{kv}$, so reducing $n_{kv}$ by a factor of $g$ reduces the cache by the same factor. Decoding reads the whole cache for every new token, so the same factor applies to memory traffic per token.

The quality cost is what the mapping gives up. Query heads in one group must use the same keys and values, so they cannot select different content from the same stored tokens. MHA gives each head its own key/value subspace. MQA, proposed by Shazeer, goes to the extreme of one shared head [corpus: library-papers/deepseek-v4-2026.pdf p.11 cites Shazeer 2019 for MQA]. GQA, by Ainslie et al. 2023, sits between MQA and MHA, with $1<n_{kv}<n_h$ [corpus: library-papers/deepseek-v4-2026.pdf p.13 cites Ainslie et al. 2023].

**Cross-layer KV sharing** is a further reduction on top of GQA. Some layers reuse the keys and values computed by an earlier layer instead of storing their own. The cache then scales with the number of layers that produce KV, not with every layer. This is the design idea only: the corpus does not contain a source for it, so the numbers in this lesson are the arithmetic, not a measured result.

## Signal vs. noise

**Signal:** the mapping $\lfloor h/g\rfloor$, the cache formula $2\,n_{kv}\,d_h$ per token per layer, the rule that $n_{kv}$ divides $n_h$, and the fact that the scale still uses $d_h$.

**Noise:** the exact recipe for converting an MHA checkpoint to GQA, and kernel-level layout details. Know that the conversion exists, and move on.

## Where it breaks

- **Modulo mapping.** Using $h \bmod n_{kv}$ instead of $\lfloor h/g\rfloor$ runs without error and gives the wrong attention. The output has the right shape, so only a value check finds it. A one-token example checks only the value mapping, since one key always gets weight 1. p01's two-token public test checks the key mapping too.
- **Sizing the cache with $n_h$.** A cache allocated for all query heads wastes a factor of $g$ and hides the saving.
- **Forgetting the scale.** The softmax scale is $1/\sqrt{d_h}$, not $1/\sqrt{d_{model}}$.
- **Cross-layer sharing with mismatched shapes.** Sharing needs the same $n_{kv}$ and $d_h$ in both layers. It also changes which layer's residual stream the keys come from, so the quality effect has to be measured, not assumed.

## Interview angle

A calm 60-second answer: multi-head attention stores $n_h$ key and value heads per token per layer, and long-context decoding is limited by reading that cache. GQA stores $n_{kv}$ heads and maps each query head to its group, so the cache shrinks by $g$. MQA is the extreme with one head. The cost is that a group shares its keys and values. Cross-layer sharing reduces the count of cached layers further. Full versions are in `interview.md`.

## Connections

This topic builds on [multi-head attention](../../dl/multi-head-attention/), whose head split and output projection p01 reuses. The next steps are the incremental decoding topic, which introduces the cache these numbers describe, and multi-head latent attention, which stores a compressed latent instead of full heads. The lineage runs from MHA (Vaswani et al. 2017), to MQA (Shazeer 2019), to GQA (Ainslie et al. 2023).
