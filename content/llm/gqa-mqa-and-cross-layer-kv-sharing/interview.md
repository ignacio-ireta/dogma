# Interview notes — GQA, MQA and cross-layer KV sharing

## Q1. Why does GQA reduce serving cost, and what does it cost in quality?
**60-second answer.** Decoding stores keys and values for every past token in every layer and reads them at every step. With $n_h$ KV heads that cache is $2\,n_h\,d_h$ values per token per layer. GQA stores $n_{kv}$ heads and maps each query head to its group, so the cache shrinks by $g=n_h/n_{kv}$ and so does the memory traffic per token. The cost is that query heads in a group must share keys and values, so they cannot each pick their own content from the same tokens. A well-chosen $n_{kv}$ keeps most of the quality at a much smaller cache.
**Follow-ups.**
- *How do you choose $n_{kv}$?* Pick the smallest value that meets the quality target at the target context length. Measure, because the answer depends on the model.
- *Why not always MQA?* It has the smallest cache, but the largest quality loss of the three.

## Q2. Walk me through the head mapping for 8 query heads and 2 KV heads.
**60-second answer.** The group size is $8/2=4$. Query heads $0..3$ read KV head $0$, and heads $4..7$ read KV head $1$. The mapping is $\lfloor h/4\rfloor$. The keys and values are expanded so each query head has its own copy of its group's head, and the rest of attention is unchanged, including the $1/\sqrt{d_h}$ scale.
**Follow-ups.**
- *What is the common bug?* Using $h \bmod n_{kv}$. The shapes are the same, so nothing errors. A one-token test catches it because the value each head outputs depends only on the mapping.
- *Does the scale change?* No. It depends on the head width $d_h$, not on the number of heads.

## Q3. What is cross-layer KV sharing and where does it fit?
**60-second answer.** It is a further reduction on top of GQA. Some layers reuse the keys and values from an earlier layer instead of storing their own. The cache then grows with the number of layers that produce keys and values, not with the total depth. It saves memory, but it changes which hidden states the shared keys come from, so you must measure the quality effect on your own model. No source in the library covers this technique yet, so cite a specific paper before quoting numbers.
**Follow-ups.**
- *What must match between the sharing layers?* The number of KV heads and the head width, so the cached tensors have the same shape.
- *How would you test it?* Compare perplexity and a task metric against the unshared model, at the target context length.

## Common mistakes
- Using the modulo mapping, or the mapping with the groups interleaved.
- Sizing the cache with $n_h$ instead of $n_{kv}$.
- Scaling scores by $\sqrt{d_{model}}$ instead of $\sqrt{d_h}$.
- Calling GQA a memory saving on the weights. It mainly reduces the cache, and the key/value projections get smaller too.
