## Hint 1 — concept

Each query scores every key with a dot product. A softmax over the keys turns one query's scores
into weights, and the output mixes the value rows with those weights. A mask removes query-key
pairs before the softmax, so they get exactly zero weight. A row with no allowed key must stay
all zero instead of turning into `NaN`.

## Hint 2 — formula / approach

Scores are `z = q @ k^T / sqrt(d_k)`. `torch.softmax` is banned here, so write the stable
softmax yourself, over the key axis only: subtract each row's maximum taken over the *allowed*
keys, exponentiate, set the disallowed entries to 0, then divide by the row sum. For a fully
masked row that sum is 0. Guard the division so the weights stay 0. Check every input rule from
the contract before you compute anything.

## Hint 3 — code shape

1. Validate the dtypes, ranks, batch shapes, sizes and the mask (boolean, broadcastable).
2. Compute the scores with `k.transpose(-2, -1)` and divide by `math.sqrt(d_k)`.
3. Broadcast the mask to the scores' shape (`None` means every pair is allowed).
4. Take the row maximum over allowed entries only. Rows with nothing allowed need a safe
   stand-in maximum.
5. Exponentiate the shifted scores where allowed and use 0 elsewhere (`torch.where`).
6. Divide by the row sums only where a sum is positive.
7. `output = weights @ v`. Return `(output, weights)`.
