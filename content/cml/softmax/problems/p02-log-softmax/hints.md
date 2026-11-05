## Hint 1 — concept

The log of a probability is the logit minus a normaliser, and the normaliser is a log-sum-exp. You never
need the probability itself, so you never need to exponentiate a large logit. Ask where the subtraction of
a constant is free, and which constant makes every exponent non-positive.

## Hint 2 — formula / approach

Let $m = \max_k z_k$ along the axis. Then $\log\sum_k e^{z_k} = m + \log\sum_k e^{z_k - m}$, and every
exponent $z_k - m$ is at most $0$. Compute the result as $(z_j-m)-\log\sum_k e^{z_k-m}$. Keeping both terms shifted avoids losing the log normalizer to a huge common offset. Only an out-of-range negative log-probability must become $-\infty$.

## Hint 3 — code shape

Validate as in softmax: reject a scalar, non-finite entries, an empty axis and an out-of-range axis. Then
keep the per-axis maximum and all reductions as axes. Store the shifted logits, compute their log normalizer, and subtract it in shifted coordinates. Opposite float64 extremes may produce a legitimate `-inf` difference; handle that subtraction without emitting an overflow warning.
