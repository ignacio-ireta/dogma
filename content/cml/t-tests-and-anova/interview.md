# Interview notes: t-tests and ANOVA

## Q1. Two groups with different sizes and possibly different variances. Which t-test do you use, and why?

**60-second answer.** Welch's t-test. It computes the standard error from each group's own variance and
uses Welch–Satterthwaite degrees of freedom, so it stays valid whether or not the spreads match. Student's
pooled test is only correct when the variances are equal, and it can give a misleading p-value when the
spreads differ, especially when the sizes differ too. Welch costs almost nothing when the variances are
equal, so it is the default. The assumption that remains is independence between the two groups.

**Follow-ups.**
- *What if the units are the same people measured twice?* Use a paired test on the differences. The
  pairing removes between-unit variation, which the two-sample test would keep in the standard error.
- *What about non-normal data?* At moderate sizes the t-test is robust to skew. With heavy tails, outliers
  or very small samples, a permutation test on the same statistic is a reasonable check.
- *The Welch test is not significant. Are the means equal?* Not shown. With equal spreads, a half-SD
  difference and 20 users per group gives power of about 0.34 at $\alpha = 0.05$. Report the interval for
  the difference, and check the size of the effect you could have missed.

## Q2. You ran a one-way ANOVA across five variants and got p < 0.05. What next?

**60-second answer.** A significant ANOVA means at least one mean differs. It does not say which, and it
does not say how large the difference is. I would look at the data for outliers and check the residuals,
then compare the pairs that matter with a multiple-comparison procedure such as Tukey's HSD, or Holm if
there are only a few planned comparisons. I would report the effect sizes with intervals, for example the
share of variance explained, and make the decision on the size of the effect, not on the p-value alone.

**Follow-ups.**
- *Why not run pairwise t-tests and stop at p < 0.05?* Each comparison adds a false positive chance. Ten
  independent pairwise tests at 5% each give about a 40% chance of at least one false positive
  ($1 - 0.95^{10}$), against 5% for a single test.
- *What does η² mean?* The fraction of total variation explained by group membership, SSB / SST.

## Q3. How is ANOVA related to a t-test?

**60-second answer.** With two groups ANOVA is the same test. The F statistic equals the square of the
pooled t statistic, and the p-values are identical. With more groups, ANOVA is variance decomposition:
the total sum of squares splits into a between-group part and a within-group part. The F ratio compares
them, scaled by their degrees of freedom. Regression with indicator variables gives the same test.

**Follow-ups.**
- *Why divide by degrees of freedom?* Each sum of squares averages over a different number of independent
  pieces. Dividing by those counts gives two comparable variance estimates.
- *What if the group variances differ a lot?* Use Welch's version of ANOVA, which adjusts the denominator.

## Common mistakes

- Using Student's pooled test by default when the spreads differ.
- Treating paired data as two independent samples.
- Reading a significant ANOVA as "all groups differ".
- Running many pairwise t-tests after ANOVA without a multiple-comparison correction.
- Computing an upper-tail F p-value as `1 - cdf`, which rounds small p-values to zero.
