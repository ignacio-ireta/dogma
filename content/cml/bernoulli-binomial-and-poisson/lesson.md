---
title: Bernoulli, binomial and Poisson
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Three count distributions form one chain. A Bernoulli trial is a binomial with $n=1$, and a binomial
becomes a Poisson when there are many trials, each with a small chance.

| Distribution | Support | pmf | Mean | Variance | Use when |
|---|---|---|---|---|---|
| Bernoulli$(p)$ | $\{0,1\}$ | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | one trial, success or failure |
| Binomial$(n,p)$ | $\{0,\dots,n\}$ | $\binom nk p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | successes in $n$ trials, $n$ known |
| Poisson$(\lambda)$ | $\{0,1,\dots\}$ | $\lambda^k e^{-\lambda}/k!$ | $\lambda$ | $\lambda$ | events in an interval, no upper count |

The binomial is a sum of $n$ independent Bernoulli$(p)$ variables. Means add for any variables, and
variances add for independent ones, so the binomial mean and variance follow from the Bernoulli values.

The Poisson is the limit of the binomial as $n\to\infty$ with $np=\lambda$ held fixed.

**Decision rule.** Fixed number of trials, same probability each: binomial. Rate over an interval with
no natural maximum: Poisson. Variance well above the mean: neither, so look for a mixture or a
dependence (see *Where it breaks*).

## Worked example

**Bernoulli to binomial.** Three independent fair coins, $n=3$, $p=\tfrac12$:
$P(k)=\binom3k(\tfrac12)^3$ gives $(\tfrac18,\tfrac38,\tfrac38,\tfrac18)$ for $k=0,1,2,3$. The probabilities
sum to 1, and the mean is $3\cdot\tfrac12=1.5$.

**Binomial moments.** $n=20$, $p=0.3$: mean $20\cdot0.3=6$, variance $20\cdot0.3\cdot0.7=4.2$, standard
deviation about 2.05. Summing the pmf directly gives the same values.

**Binomial to Poisson.** Hold $\lambda=3$ and set $p=\lambda/n$:

| $k$ | $n=10$ | $n=100$ | $n=1000$ | $n=10^6$ | Poisson(3) |
|---|---|---|---|---|---|
| 0 | 0.02825 | 0.04755 | 0.04956 | 0.04979 | 0.04979 |
| 1 | 0.12106 | 0.14707 | 0.14914 | 0.14936 | 0.14936 |
| 2 | 0.23347 | 0.22515 | 0.22415 | 0.22404 | 0.22404 |

At $n=10^6$ the largest gap over $k\le20$ is $3\times10^{-7}$. At $n=10$ it is $0.043$.

**Overdispersion.** Half of the users have rate 1 and half have rate 5. The mean is $3$. The variance is
$\mathbb E[\lambda]+\operatorname{Var}(\lambda)=3+4=7$, more than twice the mean. A Poisson(3) has
variance 3. The chance of 10 or more events is $0.016$ under the mixture and $0.0011$ under Poisson(3).

## Why it works

The binomial pmf counts the arrangements of $k$ successes among $n$ trials, $\binom nk$, and multiplies
by the probability of one arrangement, $p^k(1-p)^{n-k}$. Set $p=\lambda/n$ and let $n\to\infty$ with
$k$ fixed. Then $\binom nk\sim n^k/k!$, $p^k=\lambda^k/n^k$, and $(1-\lambda/n)^{n-k}\to e^{-\lambda}$. The
powers of $n$ cancel, leaving $\lambda^k e^{-\lambda}/k!$. Blitzstein and Hwang (*Introduction to
Probability*, the chapter on named discrete distributions) derive this and the moments.

For the moments: a Bernoulli has $\mathbb E[X]=p$ and $\mathbb E[X^2]=p$, so the variance is $p-p^2$.
For a Poisson, $\sum_k k\,\lambda^k e^{-\lambda}/k!=\lambda$, and the second moment gives variance $\lambda$.

## Signal vs. noise

- **Signal:** the moment table. "Variance equals mean" is the one-line check for Poisson counts.
- **Signal:** the binomial needs independent trials with one common $p$. Violating either is the usual
  way it fails.
- **Signal, for implementation:** $\binom{10^6}{5\cdot10^5}$ has about 301,000 digits. Never form it.
  Compute the pmf from log-gamma terms.
- **Noise for now:** the Poisson-binomial (trials with different $p_i$), the negative binomial's
  parametrisation, and generating functions.

## Where it breaks

- **Overdispersion.** Real counts often have variance well above the mean: clustered events, users who
  convert more often than others, bursty traffic. Heterogeneity in the rate adds $\operatorname{Var}(\lambda)$
  to the variance. A Poisson fit then gives standard errors that are too small and intervals that are
  too narrow.
- **Dependence.** The binomial variance $np(1-p)$ assumes independent trials. Repeated trials on one
  user are correlated, so the true variance is larger than the formula says.
- **Numerical cancellation.** $\log\Gamma(10^6+1)\approx1.3\times10^7$. Subtracting the log-gamma terms
  loses about seven digits, so a correct log-space binomial pmf at $n=10^6$ has relative error near
  $2\times10^{-9}$, not $10^{-15}$. Compare against a reference at a tolerance near $10^{-7}$.
- **The Poisson limit is slow for large $p$.** Le Cam's inequality bounds the gap between the binomial
  and Poisson by a constant times $np^2$. Small $p$ makes it small. $n=10$ with $p=0.3$ is far from it.

## Interview angle

"Binomial or Poisson?" Answer shape: a fixed number of trials with a common probability gives a
binomial. A rate over an interval with no natural maximum gives a Poisson. Then check the variance
against the mean before trusting either. Follow-ups: why the Poisson is the binomial limit, and what
overdispersion does to confidence intervals. Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/probability-basics` (independence and the product rule).
- **Unlocks:** `cml/expectation-and-variance`, `cml/beta-binomial-conjugate-updating` (a binomial
  likelihood with a beta prior), `cml/generalized-linear-models` (Poisson and binomial regression),
  `cml/sampling-distributions-standard-errors-and-cis` (standard errors of a proportion).
- **Lineage:** Jakob Bernoulli, *Ars Conjectandi* (1713), proved the law of large numbers for repeated
  trials. Siméon-Denis Poisson introduced his distribution in 1837. Bortkiewicz (1898) used it to model
  deaths from horse kicks in the Prussian army.
