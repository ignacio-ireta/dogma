# Interview notes: Sampling distributions, standard errors and CIs

## Q1. What does a 95% confidence interval mean?

**60-second answer.** The interval comes from a procedure. If I repeated the sampling many times and
built an interval each time, about 95% of those intervals would contain the fixed true parameter. For
one computed interval, the parameter is either inside it or not, so I would not say there is a 95%
probability that it is inside. The 95% describes the method's long-run coverage, and the coverage holds
only if the assumptions hold: independent observations, and a sampling distribution close to the one
the formula assumes.

**Follow-ups.**
- *So what would a Bayesian say?* A credible interval gives a probability for the parameter, given the
  prior and the data. It answers a different question.
- *Does a wider interval mean more confidence?* It means more uncertainty in the estimate. The
  confidence level is fixed by design, and the width shows the precision.

## Q2. How would you put an interval on a mean from 12 skewed observations?

**60-second answer.** With unknown σ I use a Student-t interval: $\bar x \pm t_{0.975,11}\, s/\sqrt{12}$,
where $t_{0.975,11} = 2.201$. At $n=12$ with skewed data, the sample mean is still skewed, so this
interval covers somewhat less than 95%. I would say that explicitly. A bootstrap percentile interval is
a reasonable alternative, and I would check it with a simulation that matches the data-generating
process before trusting either.

**Follow-ups.**
- *Why not the 1.96 normal quantile?* With 11 degrees of freedom the t quantile is about 12% larger.
- *What if the observations are repeated measures per user?* They are not independent. The effective
  sample size is closer to the number of users, so the standard error must be computed at that level.

## Q3. How does the standard error change with sample size, and what does that mean for planning?

**60-second answer.** The standard error is $\sigma/\sqrt n$. Halving it needs four times the data, so
precision improves slowly. For planning, I fix the width I need, estimate σ from pilot data, and solve
for n. The square root is the reason large experiments give diminishing returns in precision.

**Follow-ups.**
- *What about a metric with heavy tails?* Its σ may be large or infinite. Then the SE estimate is
  unstable, and I would use a robust statistic or a transformation.
- *Can I trade width for confidence?* Yes, at a fixed n. A lower confidence level gives a narrower
  interval, but it covers the true value less often.

## Common mistakes

- Saying there is a 95% probability that the parameter lies in one computed interval.
- Using the normal quantile for a mean with unknown σ and small n.
- Dividing the sum of squares by n instead of n − 1 when estimating σ.
- Reporting the standard error as if it were the standard deviation of the data.
- Treating repeated rows from the same user as independent observations.
