## Hint 1 — concept

Each extra sequence in a decode batch adds FLOPs and also adds KV bytes, while the weight read is shared
by the whole batch. Ask whether the FLOPs can ever catch up with the bandwidth, and how the batch enters
each of the two sides of the compute-bound test.

## Hint 2 — formula / approach

Write the compute-bound test as `2 * params * B * bandwidth >= peak * (weights + B * per_sequence)`, where
`per_sequence = (context + 1) * kv`. Both sides are linear in `B`, so move the `B` terms to one side. The result
is `B * slope >= needed`, with `slope = 2 * params * bandwidth - peak * per_sequence` and
`needed = peak * weights`. A slope that is not positive can never satisfy it.

## Hint 3 — code shape

After the validation, compute the weight bytes, the KV bytes per token and the bytes per sequence as integers.
Then:

- Return `0` when the slope is not positive.
- Otherwise take the ceiling of `needed / slope` with integer arithmetic. A float division would round large
  thresholds down.
- Return at least `1`.
