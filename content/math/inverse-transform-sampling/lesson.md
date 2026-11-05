---
title: Inverse-transform sampling
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Let $F(x) = P(X \le x)$ be the CDF of $X$, and let $U \sim \mathrm{Uniform}(0,1)$. For a
continuous, strictly increasing $F$, the draw $X = F^{-1}(U)$ has distribution $F$.

| Target | CDF | Inverse map | Code |
|---|---|---|---|
| Exponential, rate $\lambda > 0$ | $F(x) = 1 - e^{-\lambda x}$ for $x \ge 0$ | $F^{-1}(u) = -\ln(1-u)/\lambda$ | `-np.log1p(-u) / rate` |
| Discrete on $0, \dots, k-1$ | $c_i = p_0 + \dots + p_i$ (a step function) | the category $i$ with $c_{i-1} \le u < c_i$, $c_{-1} = 0$ | `np.searchsorted(cdf, u, side="right")` |

The discrete rule is a half-open interval. Category $i$ owns a stretch of $[0,1)$ of length $p_i$,
and a uniform draw lands in exactly one stretch.

## Worked example

**Exponential.** Take $\lambda = 2$ and $u = 0.5$. Then $F^{-1}(0.5) = -\ln(0.5)/2 = 0.3466$.
Check: $F(0.3466) = 1 - e^{-0.6931} = 0.5$.

**Discrete.** Take $p = (0.2, 0.3, 0.5)$, so $c = (0.2, 0.5, 1.0)$. Count the entries of $c$ that
are at most $u$. That count is the category.

1. $u = 0.35$: one entry ($0.2$) is at most $0.35$, so the category is 1. It owns $[0.2, 0.5)$.
2. $u = 0.50$: two entries are at most $0.5$, so the category is 2. The boundary belongs to the
   right-hand stretch.
3. $u = 0.04$: no entry is at most $0.04$, so the category is 0.

## Why it works

For a continuous, strictly increasing $F$, $P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)$. The
transform preserves the law exactly, with one uniform per draw and no rejection. The same identity
gives the discrete case: $P(\text{category} = i) = c_i - c_{i-1} = p_i$.

The exponential map uses $1 - u$. Since $1-U$ is also uniform, $-\ln(u)/\lambda$ is also
exponential, but it is a different map, so the same uniforms give different draws. Keep to one
map and state it.

`log1p(-u)` computes $\ln(1-u)$ without forming $1-u$ first. Every draw from `rng.random` is a
multiple of $2^{-53}$, so $1-u$ is exact there and the two forms agree. They part ways for values
off that grid. At $u = 10^{-18}$, $\ln(1-u)$ returns $0$, while $\mathrm{log1p}(-u)$ returns
$-10^{-18}$. Use `log1p` as the habit.

For the discrete case, $c$ is a step function, so each draw is a search. After an $O(k)$ cumulative
sum, each draw costs $O(\log k)$.

## Signal vs. noise

- **Signal:** $X = F^{-1}(U)$, and the half-open rule $c_{i-1} \le u < c_i$. The rule is a
  convention, but it has to be fixed before draws can be reproduced.
- **Signal:** a category with probability zero owns an empty stretch, so it is never returned.
- **Noise for now:** alias tables give $O(1)$ draws after an $O(k)$ build. They matter for
  millions of draws, not for interview problems.

## Where it breaks

- **No closed-form inverse.** The normal CDF has no elementary inverse. Use a numerical inverse
  (such as `statistics.NormalDist().inv_cdf`), a special transform, or rejection sampling.
- **Rounding at the top of the CDF.** The cumulative sum of ten $0.1$s ends at $1 - 2^{-53}$, not
  at $1$. A uniform above that value has no stretch to land in, so the index must be clipped to
  the last category.
- **Probabilities that do not sum to one.** Check the sum with a tolerance. Reject clearly wrong
  vectors, and accept sums that are off only by rounding.
- **Large exponential draws.** The largest draw from the $2^{-53}$ grid is $\ln(2^{53})/\lambda
  \approx 36.7/\lambda$. The tail is finite, but it is bounded by the resolution of the uniforms.

## Interview angle

**60-second answer.** If $F$ has a closed-form inverse, draw $u$ and return $F^{-1}(u)$. That
costs one uniform per draw, and the law is exact. For a discrete distribution, build the cumulative
sums and search for $u$. When $F^{-1}$ has no closed form, as for the gamma or beta distributions,
use rejection sampling, which draws from an easier proposal and keeps some of the draws.

**Follow-ups.**

- *Why is it correct?* For continuous $F$, $P(F^{-1}(U) \le x) = F(x)$.
- *A million categories, repeatedly?* Build the table once. Each draw is a binary search, or use an
  alias table for $O(1)$ per draw.

## Connections

- **Requires:** `cml/probability-basics`.
- **Unlocks:** `math/building-generators-by-rejection`, for the case with no closed-form inverse.
  `math/monte-carlo-simulation`, which consumes these draws.
- **Lineage:** The inversion method is a standard construction in simulation texts. Devroye (1986)
  treats it in depth, alongside rejection methods.
