# Interview notes — KV-cache size arithmetic

## Q1. How much memory does the KV cache need for a served model?
**60-second answer.** Start with the formula: two times layers times KV heads times head dimension times
tokens times batch times bytes per element. The factor two is K and V. Use KV heads, not query heads. For a
7B-class multi-head fp16 model that is 512 KiB per token, so 2 GiB per 4096-token sequence. The batch
multiplies that, and the cache eventually outgrows the weights. Name the units on every number.
**Follow-ups.** What does GQA do to the number? It divides the cache by the query-to-KV head ratio. How
does a sliding window change it? It caps the stored tokens per layer at the window size, so the cache stops
growing with context past that point.

## Q2. Given a fixed GPU memory budget, how many sequences can you serve?
**60-second answer.** Subtract the weights and any reserve from device memory. The remainder is the cache
budget. Divide by the per-sequence cache bytes and round down, because a partial sequence does not fit. If the
weights alone exceed the budget, the answer is zero, not negative.
**Follow-ups.** What if the context varies? Size for the largest context you allow, or move to paged
allocation. Why not use the average context? One long request can exhaust the budget, and the scheduler must
handle it.

## Common mistakes
- Using query heads where the formula needs KV heads, which overestimates by the group ratio.
- Dropping the factor of 2 for K and V.
- Mixing decimal GB with binary GiB in one answer.
- Applying a sliding window as a multiplier instead of a cap on stored tokens.
- Forgetting that the cache grows with generated tokens as well as the prompt.
