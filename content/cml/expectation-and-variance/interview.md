# Interview notes: Expectation and variance

## Q1. What is the variance of X + Y when X and Y are dependent?

**60-second answer.** The mean is linear, so $\mathbb{E}[X+Y] = \mathbb{E}[X] + \mathbb{E}[Y]$ with no
assumption about the joint distribution beyond finite means. For the variance, expand the square: $\operatorname{Var}(X+Y) =
\operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X,Y)$. The covariance is the only term
that depends on how X and Y move together. If they are independent, it is zero and the variances add. If
they are positively correlated, the sum is more spread out than the independent case. That is why a
portfolio's risk depends on correlations, not just on each asset's volatility.

**Follow-ups.**
- *Why does $\operatorname{Var}(X-Y)$ also add the variances?* The covariance term carries a minus sign,
  and the variances still add, because $\operatorname{Var}(-Y) = \operatorname{Var}(Y)$.
- *Does $\operatorname{Var}(X+Y)=\operatorname{Var}X+\operatorname{Var}Y$ imply independence?* No. It only
  needs zero covariance, which is weaker than independence.

## Q2. How would you compute a variance over values near $10^9$ without losing precision?

**60-second answer.** Do not use $\mathbb{E}[X^2] - \mu^2$. Both terms are near $10^{18}$, and float64
has a spacing of 128 there, so the difference can be negative. Compute the mean first and then the
average of the squared deviations, which keeps each term small. For streaming data, use a one-pass
update such as Welford's method, which also updates the mean and the sum of squared deviations
without the large-number subtraction. The fix costs one extra pass over the data, which is cheap next
to the bug it prevents.

**Follow-ups.**
- *Why is the naive formula still in textbooks?* It is correct in exact arithmetic and easy to derive.
- *Does subtracting the mean always help?* It helps with large offsets. For values of similar scale it
  makes little difference, and it costs one extra pass over the data.

## Q3. How do you decompose the variance of a mixture or a grouped sample?

**60-second answer.** Use the law of total variance: $\operatorname{Var}(X) = \mathbb{E}[\operatorname{Var}(X
\mid Z)] + \operatorname{Var}(\mathbb{E}[X \mid Z])$. The first term is the average spread inside each group.
The second is the spread of the group means. Both are non-negative, so the total is at least the average
within-group variance. This is the same decomposition behind ANOVA and the "explained versus residual"
split in regression.

**Follow-ups.**
- *When is the between-group term zero?* When every group has the same mean.
- *What does a large between-group term tell you?* The grouping variable explains much of the variation,
  so a model that ignores it leaves a large residual.

## Common mistakes

- Writing $\operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)$ without checking the
  covariance.
- Using $\operatorname{Var}(aX) = a\operatorname{Var}(X)$. The factor is $a^2$.
- Forgetting that probabilities must sum to 1, so a table that sums to 0.9 is not a distribution.
- Computing the variance as $\mathbb{E}[X^2] - \mu^2$ in floating point with large values.
- Pushing a derivative inside the expectation of a step or indicator loss, which gives a zero gradient.
