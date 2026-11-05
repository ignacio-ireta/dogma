---
title: "Incremental decoding with a KV cache"
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 with Hint 1 open."
---

## The system

Each attention layer computes queries, keys and values from its input: $Q=XW_q^\top$, $K=XW_k^\top$ and $V=XW_v^\top$, with $X\in\mathbb{R}^{B\times T\times d}$, $d$ the model width, and $h$ heads of width $d_k=d/h$. During decoding, a token's keys and values never change once computed, because the token's input does not change. So a layer stores them and appends.

A **cache** for one layer is a pair $(K_c,V_c)$ with shape $(B,h,S,d_k)$, where $S$ is the number of cached positions. One step takes $T$ new tokens and the cache, and does four things:

1. Project only the new tokens: $Q_{\text{new}},K_{\text{new}},V_{\text{new}}\in\mathbb{R}^{B\times h\times T\times d_k}$.
2. Append along the key axis: $K=[K_c;K_{\text{new}}]$ and $V=[V_c;V_{\text{new}}]$, shape $(B,h,S+T,d_k)$.
3. Attend. Query $i$ of the new block sits at absolute position $S+i$, and it may read keys $j\le S+i$.
4. Return the output $(B,T,d)$ and the new cache $(K,V)$ with $S+T$ positions.

The prefill step is the same function with $S=0$ and $T$ equal to the prompt length. Decode steps then use $T=1$.

## Worked example

Use identity weights, $d=2$, $h=1$. Step 1 has cache $\varnothing$ and input $x_0=[1,0]$. There is one key, so the softmax weight is $1$ and the output is $[1,0]$. The new cache holds $K=V=[[1,0]]$, so $S=1$.

Step 2 has input $x_1=[0,1]$. The scores are $q\cdot k_j/\sqrt{2}$ for $k_0=[1,0]$ and $k_1=[0,1]$, giving $[0,\,0.7071]$. The softmax is $[0.3302,\,0.6698]$, so the output is $[0.3302,\,0.6698]$. The query is at position 1 and sees both keys, so no mask is needed. The cache is now $S=2$.

A two-token chunk on an empty cache needs the causal rule. For $x=[[1,0],[0,1]]$, query 0 sees only key 0 and returns $[1,0]$. Query 1 sees both keys and returns $[0.3302,\,0.6698]$, the same as step 2 above.

## Why it works

The cache is exact, not an approximation. The keys and values of earlier tokens are the same whether they are recomputed or stored, since the layer is causal: a token's output depends only on tokens at or before it. So the cached step must match full recomputation over the whole prefix with a causal mask, and that check is the test for this topic.

Cost per decode step, with prefix length $T$ and width $d$. Recomputing the prefix projects $T$ tokens, about $T d^2$ multiply-adds, and attends with $T\times T$ scores, about $T^2 d$. With the cache, the projection is one token, about $d^2$, and attention is one query against $T$ keys, about $Td$. Attention drops from $O(T^2)$ to $O(T)$ per step, and projections from $O(T)$ to $O(1)$ tokens.

Memory per cached token, per sequence, is $2\cdot L\cdot d$ values for $L$ layers, times the bytes per value. For $L=32$, $d=4096$ and fp16 (2 bytes), that is $2\cdot 32\cdot 4096\cdot 2 = 524{,}288$ bytes, or 0.5 MiB per token. A 4096-token sequence then needs 2 GiB, which is why cache size limits batch size.

## Signal vs. noise

Signal: the append along the key axis, the bottom-right offset $j\le S+i$, and the rule that one new query needs no mask. Noise: the memory layout and preallocation, which are engineering details. A rotary position encoding changes what is stored but not the offset rule. Do not spend time on paged or quantised caches here.

## Where it breaks

- **Top-left alignment.** Using a causal mask of shape $(T,T)$ for a chunk with cache ignores the offset $S$. Query $i$ then sees keys $\le i$ instead of $\le S+i$, and the output is silently wrong.
- **Replacing the cache.** Storing only the new $K$ and $V$ drops the history. Each step then attends to $T$ keys, not $S+T$.
- **Mask skipped for chunks.** A single new token needs no mask. A chunk of several new tokens does, because they attend to each other.
- **Storing before projection.** Caching $X$ instead of $K$ and $V$ works, but it recomputes the projections each step and loses the saving.
- **Head layout.** Concatenating the cache along the wrong axis mixes heads and tokens. Check the shape $(B,h,S,d_k)$ before every concatenation.

## Interview angle

Sixty-second answer: a decoder layer caches its keys and values, because they are fixed once a token is processed. Each step projects only the new tokens, appends them, and attends from the new queries to every cached key, using a mask that is bottom-right aligned. A single new token needs no mask. The cache grows by $2\cdot L\cdot d$ values per token, so memory, not compute, usually limits the batch. Likely follow-ups: why not recompute, which costs $O(T^2)$ attention per step against $O(T)$ with the cache; and why `is_causal` is wrong for a cached step, since it aligns top-left.

## Connections

Prerequisites: [multi-head attention](../../dl/multi-head-attention/lesson.md) for the head layout, and [causal masking](../../dl/causal-masking/lesson.md) for the bottom-right rule. This topic follows the [autoregressive generation loop](../autoregressive-generation-loop/lesson.md), which left the prefix recomputation to be fixed here. Later work on cache size (grouped-query attention and cache compression) reduces the stored $h$ or $d$ without changing this rule.
