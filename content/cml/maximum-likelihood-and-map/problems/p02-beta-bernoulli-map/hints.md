## Hint 1: concept

The MLE is the observed proportion of heads. The MAP is the peak of the posterior, which is the likelihood times
the prior. A Beta prior adds pseudo-counts: after k heads and n-k tails, the posterior is still a Beta, with its
parameters shifted by the observed counts.

## Hint 2: formula / approach

Let $a = \alpha + k$ and $b = \beta + n - k$. The interior mode of $\mathrm{Beta}(a,b)$ is $(a-1)/(a+b-2)$, which is
valid only when $a > 1$ and $b > 1$. Otherwise the peak is on the boundary: 0 when $a \le 1$, and 1 when $b \le 1$.
Validate first: whole numbers, $n \ge 1$, $0 \le k \le n$, and a finite positive prior.

## Hint 3: code shape

Convert `successes` and `trials` to whole numbers, check `trials >= 1` and `0 <= successes <= trials`, check each
prior parameter with `math.isfinite` and `> 0`. Then `mle = k / n`, `a = alpha + k`, `b = beta + (n - k)`, and an
`if a > 1 and b > 1:` branch for the interior formula, with an `elif a <= 1:` branch returning 0.0 and an `else`
branch returning 1.0.
