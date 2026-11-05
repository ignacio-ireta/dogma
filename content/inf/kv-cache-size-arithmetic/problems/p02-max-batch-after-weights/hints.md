## Hint 1 — concept

Ask two questions in order. How much device memory is left for caches once the weights and the reserve are
placed? And how much memory does one whole sequence need? The batch is the first answer divided by the
second, rounded down. Think about what happens when the weights alone are larger than the device.

## Hint 2 — formula / approach

Let `budget = memory_bytes - weight_bytes - reserve_bytes`. If the budget is negative, return 0. Otherwise let
`per_sequence` be the KV-cache bytes of one sequence, computed as in the KV-cache bytes problem with batch 1.
The answer is `budget // per_sequence`. Two details decide the edge cases. A budget that equals
`k * per_sequence` exactly holds `k` sequences. A sliding window caps the number of stored tokens, so
`min(seq_len, window)` replaces `seq_len` in the per-sequence count.

## Hint 3 — code shape

Validate the arguments first, then compute the three steps in order. The skeleton is:

```python
budget = memory_bytes - weight_bytes - reserve_bytes
if budget < 0:
    return 0
stored = ...  # min(seq_len, window), or seq_len when there is no window
per_sequence = ...  # 2 * layers * kv heads * head dim * stored * bytes per element
return ...  # the floor of budget over per_sequence
```

Check the boundary by hand before you run the tests: with a budget of exactly three sequences the answer is 3,
and one byte less gives 2.
