---
title: Sampling distributions, standard errors and CIs
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Two distributions are easy to confuse.

| Object | What varies | Spread | Notation |
|---|---|---|---|
| Distribution of the data | one observation $X_i$ | $\sigma$ | mean $\mu$, sd $\sigma$ |
| Sampling distribution of $\bar X$ | the whole sample, redrawn | $\operatorname{SE}(\bar X) = \sigma/\sqrt{n}$ | mean $\mu$ |

For $n$ independent observations with variance $\sigma^2$, $\operatorname{Var}(\bar X) = \sigma^2/n$.
Since $\sigma$ is unknown, estimate it with the sample standard deviation
$s = \sqrt{\tfrac{1}{n-1}\sum_i (x_i - \bar x)^2}$ (divide by $n-1$) and use $\widehat{\operatorname{SE}} = s/\sqrt{n}$.

A two-sided $(1-\alpha)$ confidence interval for $\mu$ is

$$\bar x \pm t_{1-\alpha/2,\;n-1}\cdot \frac{s}{\sqrt n},$$

where $t_{q,\nu}$ is the $q$-quantile of Student's $t$ distribution with $\nu$ degrees of freedom.

**Decision rule:** for a mean with unknown $\sigma$, always use the $t$ quantile with $n-1$ degrees of
freedom. Use $z$ only when $\sigma$ is genuinely known. The gap matters at small $n$: $t_{0.975,4} = 2.776$
against $z = 1.960$, and $t_{0.975,29} = 2.045$ at $n = 30$.

**What 95% means:** the 95% describes the procedure. Repeat the sampling many times, build an interval
each time, and about 95% of the intervals contain the fixed true $\mu$. For one computed interval, $\mu$
is either inside it or not. Saying "there is a 95% probability that $\mu$ is in this interval" is the
standard misreading. A test's size has the same procedure-level reading: $\alpha$ bounds how often a true
null is rejected in repeated experiments, not the probability that the null is true (Neyman and Pearson
1933, per the Library note).

## Worked example

Data $x = (1, 2, 3, 4, 5)$, so $n = 5$.

1. $\bar x = 3$.
2. $s^2 = (4 + 1 + 0 + 1 + 4)/4 = 2.5$, so $s = 1.5811$.
3. $\widehat{\operatorname{SE}} = 1.5811/\sqrt{5} = 0.7071$.
4. $t_{0.975,4} = 2.7764$, so the margin is $2.7764 \times 0.7071 = 1.9632$.
5. Interval: $(1.0368,\ 4.9632)$.

With $z$ instead, the margin is $1.96 \times 0.7071 = 1.386$, about 30% narrower. With normal data and $n = 5$, that
interval covers the true mean $2F_4(1.96) - 1 = 87.8\%$ of the time, not 95%, where $F_4$ is the
Student-t CDF with 4 degrees of freedom.

## Why it works

If the data are normal, $(\bar X - \mu)/(\sigma/\sqrt n)$ is exactly standard normal. Replace $\sigma$ with
$s$ and the ratio $(\bar X - \mu)/(s/\sqrt n)$ follows Student's $t$ with $n-1$ degrees of freedom. That
was Gosset's 1908 result. The heavier tails of $t$ pay for the extra uncertainty in $s$. As $n$ grows, $t$
approaches $z$.

For other data, the central limit theorem says $\bar X$ is approximately normal for large $n$, so the
interval is approximately valid. "Approximately" carries the whole caveat (see Where it breaks).

Why divide by $n-1$: the sum of squares has expectation $(n-1)\sigma^2$, so dividing by $n-1$ gives an
unbiased estimate of $\sigma^2$. Dividing by $n$ biases it low by the factor $(n-1)/n$.

## Signal vs. noise

- **Signal:** $\operatorname{SE} = \sigma/\sqrt n$. Halving the standard error needs four times the data.
- **Signal:** the quantile depends on $n$ through the degrees of freedom. Small samples widen the interval
  more than the $z$ rule suggests.
- **Signal:** the 95% is a statement about repeated sampling, not about one interval.
- **Noise for now:** exact intervals for other statistics such as medians or ratios. Resampling covers
  those (see `cml/bootstrap-and-permutation-resampling`).

## Where it breaks

- **Skew at small $n$.** For exponential data with true mean 1, the 95% $t$-interval covers the mean
  88.3% of the time at $n=5$, 90.1% at $n=10$, 92.7% at $n=30$ and 94.2% at $n=100$ (400,000 simulated
  samples per row, `numpy.random.default_rng(20261024)`). The sample mean keeps the skew of the data
  until $n$ is large.
- **Infinite variance.** A Pareto distribution with shape 1.5 has infinite variance, so $\sigma$ and the
  standard error are not finite quantities and the interval is not meaningful.
- **Dependence.** Repeated measures per user or autocorrelated time series have fewer effective samples
  than rows. Using $n$ rows makes the standard error too small.
- **Numerics.** The one-pass formula $(\sum x_i^2 - n\bar x^2)/(n-1)$ returns variance $0.0$ for
  $x = 10^9 + (0.1, -0.1, 0.2, -0.2, 0)$, while the two-pass value is about $0.025$ (the inputs carry
  rounding error near $10^{-7}$ at this magnitude). Subtract the mean first.
- **Zero spread.** A constant sample gives a zero-width interval. The arithmetic is correct, but it
  usually means the metric did not vary, which is worth checking before you report it.

## Interview angle

**60-second answer.** The sampling distribution of $\bar x$ has standard deviation $\sigma/\sqrt n$. I
estimate $\sigma$ with $s$, so I use the $t$ quantile with $n-1$ degrees of freedom. The interval is a
procedure that covers the true mean in 95% of repeated samples. It does not give a 95% probability for
this one interval. I check independence and how skewed the data are, because the normal approximation
is what the coverage rests on.

**Follow-ups.**
- *Twelve skewed observations?* The $t$ interval undercovers at this size (see Where it breaks). I would
  use a bootstrap interval and report that it is approximate.
- *Why not 1.96?* At $n=12$, $t_{0.975,11} = 2.201$, about 12% wider than $z$.

## Connections

- **Requires:** `cml/descriptive-statistics` (the sample standard deviation), `cml/probability-basics`.
- **Unlocks:** `cml/bootstrap-and-permutation-resampling` (when the formula fails), `cml/t-tests-and-anova`
  (the same $t$ distribution applied to differences), `cml/a-b-test-design-and-analysis`.
- **Lineage:** Student (W. S. Gosset), *The probable error of a mean* (1908), derived the $t$ distribution
  for the mean of normal samples.
