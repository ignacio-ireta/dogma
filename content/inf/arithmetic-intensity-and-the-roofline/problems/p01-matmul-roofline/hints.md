## Hint 1 — concept

A product is limited by whichever resource runs out first. Count the work (FLOPs) and the traffic (bytes) for
the operation, then ask which one a device delivers more slowly relative to the work. The ridge point is the
ratio where the two limits meet.

## Hint 2 — formula / approach

Count with integers first: FLOPs are `2 * m * n * k`, and bytes are the three matrices (`A`, `B` and the output
`C`) times `element_bytes`. Intensity is FLOPs divided by bytes. Decide the bound with the cross-multiplied form,
`flops * bandwidth >= peak_flops * bytes`, so a tie at the ridge is exact. The time floor is the larger of
FLOPs over peak and bytes over bandwidth.

## Hint 3 — code shape

Validate the six integers first and raise `TypeError` before `ValueError`. Then compute, in this order: the two
integer counts, the boolean bound, the attainable rate as a float, the time floor, and the returned dict:

- Check each argument with `type(...) is int`, then `>= 1`.
- Compute `flops` and `traffic` as integers.
- Set `compute_bound` from the cross-multiplied comparison.
- Return the six keys, with `"bound"` set to `"compute"` or `"memory"`.
