# Interview notes: Covariance and correlation

## Q1. What is the difference between covariance and correlation, and when do you normalise?

**60-second answer.** Covariance is the average product of two variables' centred values. Its units are the
product of the two units, so its size depends on the scale of each variable. Correlation divides by both
standard deviations. It is unitless, lies in $[-1, 1]$, and is comparable across pairs of variables. Use
correlation to ask whether two variables move together and how tightly. Use covariance when the scale matters,
for example when you need the variance of a weighted sum.

**Follow-ups.**
- *Is a covariance matrix always positive semi-definite?* Yes. For any $\mathbf{v}$, $\mathbf{v}^\top C\mathbf{v}
  = \operatorname{Var}(X\mathbf{v}) \ge 0$. It is singular when two columns are collinear.
- *Does the choice of $n$ or $n-1$ change the correlation?* No. The divisor cancels.

## Q2. Pearson or Spearman, and why might they disagree?

**60-second answer.** Pearson measures linear association and is sensitive to outliers. Spearman is the Pearson
correlation of the ranks, so it measures monotone association and ignores the size of extreme values. They
disagree when the relation is monotone but curved, as with $y = x^3$, where Pearson is $0.943$ and Spearman is
$1$, or when one point dominates. Report both when the answer depends on the choice.

**Follow-ups.**
- *Can Pearson be zero when the variables are dependent?* Yes. For $y = x^2$ with $x$ symmetric around zero, the
  covariance is zero.
- *How does Spearman handle ties?* It assigns average ranks to tied values.

## Q3. How do you check a correlation before you trust it?

**60-second answer.** Plot the data first. Anscombe's quartet has four datasets with the same means, variances and
correlation to two decimals, and only the plot shows that three of them are not linear clouds. Then check for
outliers and leverage points, for a constant column, which makes the correlation undefined, and for subgroups
that reverse the sign (Simpson's paradox). A single high correlation is evidence of association, not of
mechanism.

**Follow-ups.**
- *What does a correlation of 0.8 mean for prediction?* It says the linear fit explains about 64% of the variance
  ($r^2$), not that the relation is causal.
- *Why report a confidence interval?* A sample correlation from few points is noisy, and the interval shows
  whether the sign is reliable.

## Common mistakes

- Reading a correlation of zero as independence. Zero covariance does not imply independence.
- Comparing covariances across variables with different units, instead of correlations.
- Computing the correlation of a constant column without checking for a zero standard deviation.
- Treating correlation as causation.
