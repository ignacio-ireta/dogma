## Hint 1 — concept

A router is a softmax over experts, followed by a choice. Each token keeps only its best `k`
experts, and the kept gates are re-normalised so that they sum to 1. Two details are easy to get
wrong: which experts count as "best" when logits tie, and what gets counted for the load.

## Hint 2 — formula / approach

For token `t`, rank the experts by logit, largest first, and break ties by the lower index. Keep
the first `k`. The gate of a kept expert is its softmax probability divided by the sum of the kept
probabilities. That equals a softmax over the `k` kept logits, which is numerically safer. Count
every kept pair, so the counts sum to `T * k`. Work with logits rather than probabilities when you
rank. Two experts far below the top can both have probabilities that underflow to exactly zero,
and then their order is lost in the probabilities but not in the logits.

## Hint 3 — code shape

Sort each row of `logits` in descending order with a stable sort. The first `k` columns of the
sorted indices are `expert_indices`. Use `torch.gather` on `logits` with those indices to get the
chosen logits, and apply `torch.softmax` over the last dimension to get `gate_weights`. Flatten
`expert_indices` and pass it to `torch.bincount` with `minlength=E` to get the counts, then convert
the result to int64. Validate `k` and `logits` before any of that.
