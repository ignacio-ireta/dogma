## Hint 1 — concept

The sampler never sees the logits directly. It sees one probability vector, which is built in
stages: first the temperature reshapes the scores, then top-k removes low-ranked tokens, then
top-p removes the tail of the ranking. After each removal the rest must be rescaled so the mass
sums to one. Each stage is a mask followed by a renormalisation.

## Hint 2 — formula / approach

Compute `z = logits / T`, subtract `z.max()`, exponentiate, and divide by the sum. Sort the
probabilities descending with `stable=True` so ties keep index order. For top-k, keep the first `k`
indices of that order. For top-p, take `cumsum` of the sorted probabilities, find the first position
where it is `>= top_p`, and keep the tokens up to and including that position. After each mask,
set the removed entries to 0 and divide by the total. Use the boundary as written: `>=`, not `>`.

## Hint 3 — code shape

Write the function as a straight line of stages: validate, `z = logits / t`, the stable softmax,
`order = torch.sort(p, descending=True, stable=True).indices`, then one `if top_k` block and one
`if top_p` block. Each block builds a boolean mask with `torch.zeros(V, dtype=torch.bool)` and
sets `mask[order[...]] = True`. Apply it with `torch.where(mask, p, zeros)` and renormalise.
