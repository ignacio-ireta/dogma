# Interview notes — GPU memory and capacity sizing

## Q1. How many GPUs does a model need to serve a given load?
**60-second answer.** Start from the per-GPU share of one replica. The weights divide by the tensor-parallel degree, rounded
up. The KV cache divides by the degree through the KV heads, and the workspace stays fixed. Subtract the reserve from the
memory first. Then try the powers of two that divide the KV heads, and take the smallest one whose total fits. The
leftover over the per-sequence cache gives the concurrency. State the load, for example 32 sequences of 4096 tokens, and the
units.
**Follow-ups.** Why must the degree divide the KV heads? Each KV head lives on one GPU, so a degree that does not divide
them leaves an uneven split. What if nothing fits? Quantise the weights or the cache, or check the context length you
plan for.

## Q2. What changes the GPU count the most?
**60-second answer.** The batch and the context length, because the cache grows with both. The weights set the floor: if
they alone exceed the usable memory of the largest allowed degree, no load fits. Then the cache sets how many sequences fit
on the group you chose.
**Follow-ups.** Does more tensor parallelism always help? It shrinks the shares, but it adds communication on every layer,
and the reserve and workspace do not shrink. Why does grouped-query attention matter here? Fewer KV heads cap the degree.

## Common mistakes
- Mixing GiB and GB in one answer.
- Dividing the weights by the degree but leaving the cache whole, or the reverse.
- Splitting the cache by query heads instead of KV heads.
- Quoting a GPU count without the batch and context it was sized for.
- Treating a fitting configuration as proof of throughput.
