## Hint 1 — concept

A GPU in a tensor-parallel group holds a share of everything: a slice of each weight matrix and the KV heads it
serves. Ask what a share costs for each part, and what the GPU can actually use after a reserve. The answer is the
smallest group size whose share of the weights plus its share of the cache plus its workspace fits in that budget.

## Hint 2 — formula / approach

Compute the usable bytes first: memory times the unreserved percent, floored. Then, for each allowed degree, add three
per-GPU terms: the weight bytes divided by the degree, rounded up; the KV bytes for all the tokens of all sequences,
divided by the degree; and the workspace. The degrees are powers of two, and a degree is allowed only when it divides the
number of KV heads. The first allowed degree whose total is at most the usable bytes is the answer.

## Hint 3 — code shape

After the validation, compute the usable bytes, the stored tokens per sequence (capped by a window when one is set), and the
total weight bytes as integers. Then loop over the degrees in increasing order:

- Skip a degree that does not divide the KV heads.
- Compute the per-GPU weights with a ceiling division, and the per-GPU KV bytes from the KV heads divided by the degree.
- Return the degree when the total is at most the usable bytes.
- Return `0` when the loop ends without a fit.
