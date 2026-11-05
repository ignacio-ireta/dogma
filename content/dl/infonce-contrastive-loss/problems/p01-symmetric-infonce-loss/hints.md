## Hint 1 — concept

The loss is a cross-entropy with N candidates per anchor. The true partner of each anchor sits on the diagonal of the similarity matrix, and the other entries of its row are the in-batch negatives. Normalise first, so only directions matter. Then ask the same question in both directions: for each `a` find its `b`, and for each `b` find its `a`.

## Hint 2 — formula / approach

Let `a_hat` and `b_hat` be the rows divided by their Euclidean norms. The logits are `S = a_hat @ b_hat.T / temperature`. For each row i the loss is `logsumexp_j S[i, j] - S[i, i]`, and for the other direction you apply the same formula to `S.T`. The result is the average of the two row means. Compute the log-sum-exp by subtracting the row maximum first, so a small temperature cannot overflow.

## Hint 3 — code shape

Write one helper that takes an (N, N) logits matrix and returns the mean over rows of `log_sum_exp - diagonal`, where the diagonal comes from indexing with `arange(N)` twice. Call it on `logits` and on `logits.T`, then average. Validate the shapes, the finite values, the zero norms and the temperature before any arithmetic.
