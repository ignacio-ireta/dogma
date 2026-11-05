## Hint 1 — concept

The loss has two parts. One is how much router probability each expert receives on average. The
other is how many tokens each expert actually receives. Their product, summed over experts, is
small only when both are spread evenly. Only the probability part can carry a gradient, because
counting tokens is a discrete step.

## Hint 2 — formula / approach

Compute the softmax of each row of logits. Average each column to get `P`, of shape `(E,)`. For
`f`, find the top-1 expert of each token, breaking ties toward the lower index, count how many
tokens chose each expert, and divide by `T`. The loss is `alpha * E * sum(f * P)`. Check the
inputs first: rank, dtype, `T` and `E`, finiteness, and the type and sign of `alpha`.

## Hint 3 — code shape

Use `torch.softmax(logits, dim=-1)` and then `.mean(dim=0)` for `P`. For the top-1 choice, sort each
row with `torch.sort(..., descending=True, stable=True)` and take column 0. Pass those indices to
`torch.bincount(..., minlength=E)`, convert to float64 and divide by `T`. Return `alpha * E` times
the sum of the elementwise product of `f` and `P`. That is a 0-dimensional tensor.
