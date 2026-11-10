---
title: Bootstrap and permutation resampling
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Both methods replace a formula with a computer simulation built from the data itself.

| Method | Question | What is resampled | Output |
|---|---|---|---|
| Bootstrap | How much would the statistic move between samples? | $n$ values drawn **with** replacement | an interval for $\theta$ |
| Permutation test | Is the observed difference bigger than chance? | the pooled values, relabelled at random | a p-value |

**Bootstrap (plug-in principle).** Replace the unknown distribution $F$ by the empirical distribution $\hat F$.
Draw $B$ resamples of size $n$ with replacement, compute the statistic on each, and take the percentile
interval $[\,q_{\alpha/2},\ q_{1-\alpha/2}\,]$ of the replicates. This assumes the observations are i.i.d.
draws from one population (Efron 1979, per the Library note). Dependent or clustered rows violate it.

**Permutation test.** Under the null that both groups come from one distribution, the labels are
exchangeable. Relabel the pooled values at random many times, and count how often the relabelled difference
in means is at least as large as the observed one.

**Decision list:**
- Smooth statistic, reasonable i.i.d. $n$: percentile bootstrap. Extremes, skewed data at small $n$ or heavy
  tails: check the interval by simulation first.
- Dependent rows (time series, repeated measures): block or cluster resampling, also checked by simulation.
- "Does the label matter?": permutation test, with labels exchangeable under the null (for paired data,
  permute within pairs).

## Worked example

Two groups, $a = (1,2,3)$ and $b = (7,8,9)$. Of the $\binom{6}{3} = 20$ equally likely splits under the null,
only the observed split and its mirror reach a difference of 6 or more. The exact two-sided p-value is
$2/20 = 0.10$. Random relabelling estimates that fraction, and the $+1$ correction gives
$p = (\text{count} + 1)/(B + 1)$.

Bootstrapping the maximum of $n = 5$ values: a resample contains the sample maximum at least once with
probability $1 - (4/5)^5 = 0.672$, so the replicates are mostly ties at the top.

## Why it works

The bootstrap works when $\hat F \approx F$ for large $n$ and the statistic is smooth, so the spread of
$\theta(\hat F^*)$ mimics the spread of $\hat\theta$.

The permutation test works because under the null every relabelling was equally likely to produce the data.
The operative requirement is exchangeability, which is weaker than independence: i.i.d. draws imply it, and so
does sampling without replacement from a finite population, whose draws are not independent. The observed
statistic's rank among the relabellings is then a valid p-value.

The $+1$ matters. The observed labelling is one of the possible outcomes, so $(c+1)/(B+1)$ is never zero and
stays valid for any finite $B$. Phipson and Smyth (2010) make this argument for random permutations (cited,
not checked against the paper).

## Signal vs. noise

- **Signal:** what is resampled (with replacement for the bootstrap, relabelling for the permutation test),
  the i.i.d. premise, the $+1$, and float ties (below).
- **Noise for now:** BCa and bootstrap-$t$ refinements of the same idea.

## Where it breaks

- **Dependence.** Resampling single rows from a correlated series understates the spread. Take an AR(1)
  series with $\varphi = 0.9$, unit innovations, a stationary start and $n = 100$. The true standard error of
  the mean is 0.951 (exact, from the autocovariance sum). In one seeded run (2,000 series, $B = 1{,}000$, seed
  20261031) the i.i.d. bootstrap gives 0.20, about 0.21 of the truth, and its 95% percentile interval covers
  33% of the time. Its large-$n$ ratio is $\sqrt{(1-\varphi)/(1+\varphi)} \approx 0.23$. Moving blocks of
  length 20 give 0.57 and 69%: better, still short. Block length needs a simulation check.
- **Skew at small $n$.** For Exp(1) data with $n = 5$, the 95% percentile interval covers the mean 79.8% of
  the time (one seeded run, 4,000 series, $B = 2{,}000$), against 88.3% for the $t$-interval at this size.
- **Extremes and tiny samples.** A replicate of the maximum cannot exceed the sample maximum, so for
  Uniform(0, 1) data the interval never contains the true maximum 1. With $n = 5$ there are only
  $\binom{9}{5} = 126$ distinct resamples, so the endpoints are coarse.
- **Exchangeability.** Paired designs need within-pair permutations, not pooled ones.
- **Float ties.** On $a = (0.1, 0.5, 0.1)$, $b = (0.3, 0.3, 0.5, 0.3)$, a plain `>=` gives $p = 0.29$, while
  counting the ties gives $p = 0.63$. Use a relative tolerance near $10^{-9}$. A loose tolerance counts
  near-ties that are not ties.
- **Repeats are rows.** The 150 `iris` sepal lengths hold only 35 distinct values. Resampling distinct values instead of rows gives a 90% interval for the mean of [5.873, 6.155], against [5.727, 5.957] from the protocol (B = 2,000, seed 13). Both ends move up by 0.15 to 0.20. Float32 storage moves the protocol interval by only $1.4 \times 10^{-8}$, but a test at $10^{-12}$ sees it.

## Interview angle

**60-second answer.** For uncertainty in a statistic with no closed form, I bootstrap: resample with
replacement, recompute, and take the replicate quantiles. For a difference, I permute the labels under the
null and count how often a difference at least as large appears, with the $+1$ correction. Each method needs
one assumption stated up front: i.i.d. rows, or exchangeable labels under the null. Each fails in a known
place: extremes, tiny or skewed samples and dependence for the bootstrap, and pairs or clusters for the
permutation test.

**Follow-ups.**
- *Why $(c+1)/(B+1)$?* The observed labelling is one of the outcomes, so the p-value is never zero and
  stays valid for finite $B$.
- *Bootstrap the maximum?* No. The replicates cannot exceed the sample maximum, and about 63% of them
  equal it (for large $n$, $1 - 1/e$).

## Connections

- **Requires:** `cml/sampling-distributions-standard-errors-and-cis` (standard errors and coverage).
- **Unlocks:** `cml/t-tests-and-anova` (the permutation alternative to a t-test), `cml/chi-square-tests`,
  `cml/multiple-testing-corrections` (permutation-based error control).
- **Lineage:** Fisher's randomisation test (1935) is the root of permutation inference. Efron (1979, per the
  Library note) introduced the bootstrap.
