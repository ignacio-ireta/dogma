## Hint 1 — concept

A draw from a discrete distribution can be done with one uniform number. Lay the tokens on the
interval `[0, 1)` as consecutive segments, each as long as its probability. A uniform point lands in
one segment, and that segment's token is the draw. The running total of the probabilities gives the
segment boundaries.

## Hint 2 — formula / approach

Compute `cdf = cumsum(probs)`. Draw `u` in `[0, 1)` with `torch.rand(1, generator=..., dtype=...)`.
Return the first index with `cdf > u`. A zero-probability entry repeats the previous cdf value, so
it does not own a segment. Check the rules before drawing: the rank and dtype, the finite and
non-negative entries, the sum, and the generator type.

## Hint 3 — code shape

Use `torch.searchsorted(cdf, u, right=True)` on the one-element tensor `u`, and convert it with
`int(... .item())`. Keep a fallback for an index equal to `V`, which rounding can produce: return
the last index with `probs > 0`. Pass `generator=generator` to every random call. The random state
of the default generator must not change.
