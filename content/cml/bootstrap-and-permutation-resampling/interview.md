# Interview notes: Bootstrap and permutation resampling

## Q1. How would you put a confidence interval on a statistic with no closed form?

**60-second answer.** I would bootstrap. I resample the sample with replacement, compute the statistic
on each resample, and take the lower and upper percentiles of those replicates. The method uses only the
data, so it works for medians, ratios and model-based metrics. It assumes a smooth statistic and i.i.d.
observations from one population, so dependence such as a time series breaks it. I would check that
with a simulation of a known process before trusting the interval for a small sample.

**Follow-ups.**
- *How many resamples?* A few thousand is common. Monte Carlo error in the endpoints shrinks with the
  square root of the number of resamples, so more resamples help only up to the point where the
  resampling noise is small next to the sampling noise.
- *Would you trust it for the 99th percentile of latency?* Not at small $n$. Extreme quantiles need a lot
  of data, and the replicates will be dominated by a few values.
- *Is the percentile interval fine for skewed data at $n = 5$?* Not reliably. For Exp(1) data it covered
  79.8% in simulation, against 88.3% for the $t$-interval. I would use a larger sample or a bias-corrected
  method, and report the coverage check.

## Q2. How do you test whether a treatment changed a metric when the samples are small and skewed?

**60-second answer.** A permutation test on the difference in means. Under the null that both groups
come from one distribution, the labels are exchangeable. I pool the values, relabel them at random many
times, and count how often the relabelled difference is at least as large as the observed one. The
p-value is $(\text{count} + 1)/(B + 1)$. It needs only exchangeability under the null, which i.i.d. data
supply, and no normality assumption. It is exact when I enumerate every relabelling. For a paired design I would permute within pairs instead.

**Follow-ups.**
- *What does a permutation test not tell you?* The size of the effect. Report a confidence interval for
  the difference alongside the p-value.
- *Why does a float comparison matter?* Exact ties in the statistic can round to a value just below the
  observed one. A relative tolerance counts them, which changes p-values on small samples with repeated
  values.

## Q3. When does the bootstrap fail?

**60-second answer.** It fails in four situations. The statistic depends on extremes, such as the
maximum: every replicate is at most the sample maximum. The sample is tiny, so the replicates take few
distinct values. The observations are dependent, such as time series or repeated measures, so resampling
rows treats correlated data as independent. The distribution has heavy tails, so the replicate
distribution is unstable. In each case I would switch to a model-based method, block or cluster
resampling, or a different estimand.

**Follow-ups.**
- *Can you fix the max with a bigger sample?* No. The problem is structural, since the replicates cannot
  exceed the sample maximum.
- *What do you do for time series?* Resample blocks of consecutive observations to keep the dependence.

## Common mistakes

- Bootstrapping the sample maximum or minimum and quoting the interval as 95%.
- Resampling without replacement when a bootstrap is meant. That gives the original sample back.
- Reporting `count / B` as a p-value, which can be exactly zero.
- Comparing floating-point statistics for equality when counting ties.
- Permuting paired data as if the two groups were independent.
