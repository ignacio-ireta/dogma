# Interview notes: Bernoulli, binomial and Poisson

## Q1. Binomial or Poisson? How would you decide for a count?

**60-second answer.** Ask what the count is out of. If there is a fixed number of trials $n$, each
with the same success probability $p$, and the trials are independent, the count is binomial. For
example, conversions out of $n$ impressions. If there is no natural upper count, only a rate over an
interval, such as clicks per minute or defects per wafer, the count is Poisson with
$\lambda=\text{rate}\times\text{interval}$. Then check the data. Binomial variance is $np(1-p)$,
which is always below the mean. Poisson variance equals the mean. A sample variance well above the
mean means neither model fits without changes.

**Follow-ups.**
- *What if $n$ is large and $p$ is small?* Then the binomial and Poisson agree closely, so either is
  acceptable, and the Poisson is cheaper to fit.
- *What if trials differ in their probability?* The count is Poisson-binomial, with variance
  $\sum_i p_i(1-p_i)$, which is smaller than the binomial with the same mean.

## Q2. Why is a Poisson the limit of a binomial, and when does the approximation break?

**60-second answer.** Take $n$ trials with $p=\lambda/n$. As $n$ grows, $\binom nk\approx n^k/k!$,
$(1-\lambda/n)^{n-k}\to e^{-\lambda}$, and the binomial pmf converges to $\lambda^k e^{-\lambda}/k!$.
The approximation is good when $n$ is large and $p$ is small. Le Cam's inequality bounds the gap by a
constant times $np^2$, which is small when $p$ is small. It breaks when the trials are dependent or
their probabilities differ, which is when a Poisson-binomial or an overdispersed model is needed.

**Follow-ups.**
- *Give a number.* With $\lambda=3$, $n=10$ gives $P(0)=0.028$ against $0.050$ for the Poisson. At
  $n=10^6$ the largest gap over $k\le20$ is $3\times10^{-7}$.
- *Why does $\lambda$ equal the mean in both?* $np=\lambda$ by construction, and the limit keeps the
  mean fixed.

## Q3. How do you check count data for overdispersion, and what does it break?

**60-second answer.** Compare the sample variance with the sample mean. For Poisson data the ratio is
about 1. A ratio well above 1 is overdispersion. The usual cause is heterogeneity: units differ in
their rates, so the pooled variance includes the variance of the rates. For a half-and-half mixture of
rates 1 and 5, the mean is 3 and the variance is 7. A Poisson fitted to such data gives standard
errors that are too small, so confidence intervals are too narrow and tests are too eager to reject.
Remedies: a negative binomial model, a quasi-Poisson dispersion parameter, or cluster-robust standard
errors when the overdispersion comes from repeated observations of the same unit.

**Follow-ups.**
- *How large is the tail error?* For that mixture, $P(X\ge10)$ is $0.016$ against $0.0011$ under
  Poisson(3), about fourteen times more tail mass.
- *Is overdispersion always heterogeneity?* No. Clustering (trials within one user being correlated)
  also inflates the variance of a binomial count.

## Common mistakes

- Using a binomial when the number of trials is not fixed, such as counts per hour with no maximum.
- Fitting a Poisson without checking that the mean and the variance are close.
- Computing $\binom nk$ as a float for large $n$. It overflows, and the result is inf or NaN.
- Treating repeated observations of one user as independent trials, which understates the variance.
- Forgetting that the binomial pmf is zero outside $0..n$, and returning an error or a NaN there.
