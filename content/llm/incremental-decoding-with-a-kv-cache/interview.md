# Interview notes: Incremental decoding with a KV cache

## Q1. What does a KV cache store, and how does one decode step use it?

**60-second answer.** Each layer stores the keys and values of every earlier token, because they depend only on that token and do not change. A step projects only the new tokens, appends their keys and values to the cache, and lets the new queries attend to every cached key. The mask is bottom-right aligned, so a query at absolute position S + i sees keys up to that position. A single new token needs no mask at all. The output is the same as recomputing the whole prefix, but the work per step is one query rather than the full prefix.

**Follow-ups.**

- What does caching save? Attention per step falls from O(T^2) to O(T), and the projections from T tokens to one.
- What does it cost? Memory: 2 times L times d values per token. For a 32-layer, 4096-width fp16 model, that is 0.5 MiB per token, so long sequences and large batches compete for memory.

## Q2. How would you check a cached attention implementation?

**60-second answer.** Compare it against full recomputation over the whole prefix with a causal mask, across several steps and a prefill chunk. The cached output must match exactly, up to floating-point tolerance, at every step. The common bugs are a wrong mask offset, a cache that replaces rather than appends, and a mask skipped for multi-token chunks. Each one shows up as a mismatch in the first step where the cache is non-empty.

**Follow-ups.**

- Why does the test need several steps? A bug in the offset is invisible when the cache is empty, because then the offset is zero.
- Why not use the library's causal flag? It aligns the mask top-left for non-square shapes, so it is wrong for a query that follows a cache.

## Common mistakes

- Using a top-left causal mask for a chunk with a cache.
- Caching the layer input and recomputing the projections each step.
- Forgetting that a multi-token chunk still needs a mask among its own tokens.
- Replacing the cache instead of appending to it.
