## Hint 1 — concept

Binary cross-entropy is the negative log-likelihood of a Bernoulli model with probability $\sigma(z)$. Written
through probabilities it fails at large $|z|$, because one of the two logs is then the log of zero. Rewrite the
loss in terms of $z$ alone, so that no probability has to be formed.

## Hint 2 — formula / approach

For each example, $\ell = \log(1 + e^{z}) - yz$. Split $\log(1 + e^{z})$ as $\max(z, 0) + \log(1 + e^{-|z|})$, so
that every exponential has a non-positive exponent. The mean over examples is the answer. Targets $y$ may be
any value in $[0, 1]$, because the formula is linear in $y$.

## Hint 3 — code shape

Validate: both inputs 1-D, the same length, `z` non-empty, `z` finite, `y` finite with every value in $[0, 1]$.
Build the per-example losses with a non-positive exponential and `log1p`. For positive logits, factor the linear term as `z * (1 - y)` to avoid cancelling large terms when a soft target is near one. Before averaging large losses, scale them by their maximum to avoid overflowing their sum. Handle an all-zero loss vector separately.
