---
title: t-tests and ANOVA
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Both tests compare group means by asking how large the difference is relative to its standard error.

| Test | Statistic | Degrees of freedom | Use when |
|---|---|---|---|
| Welch two-sample | $t = (\bar a - \bar b)/\sqrt{s_a^2/n_a + s_b^2/n_b}$ | Welch–Satterthwaite (non-integer) | two independent groups, any sizes and spreads |
| Student pooled | same numerator, pooled $s_p$ | $n_a + n_b - 2$ | equal variances, assumed rather than checked |
| Paired | $t = \bar d/(s_d/\sqrt n)$ on differences $d_i$ | $n - 1$ | the same units measured twice |
| One-way ANOVA | $F = \mathrm{MSB}/\mathrm{MSW}$ | $(k-1,\; N-k)$ | $k \ge 2$ independent groups |

**Welch–Satterthwaite:** with $v_a = s_a^2/n_a$ and $v_b = s_b^2/n_b$,

$$\nu = \frac{(v_a + v_b)^2}{v_a^2/(n_a-1) + v_b^2/(n_b-1)}.$$

**Default:** use Welch. Its standard error is right whether or not the spreads match; its degrees of freedom
are an approximation. Student's test needs equal spreads, and its pooled $s_p$ is wrong when they differ.

**ANOVA:** $\mathrm{SST} = \mathrm{SSB} + \mathrm{SSW}$. Between-group variation (SSB) is divided by $k-1$, and
within-group variation (SSW) by $N-k$. The F ratio compares the two.

**Errors:** $\alpha$ is fixed in advance: the long-run rate of rejecting a true null. Power, the chance of
rejecting a real difference, depends on that difference, $n$ and the spread. Neyman and Pearson (1933, per
the Library note) frame the test as a decision with these two error rates.

## Worked example

Groups $a = (1,2,3,4,5)$ and $b = (2,4,6,8,10,12,14)$.

- Means $3$ and $8$, difference $-5$.
- Sample variances $2.5$ and $18.67$. Divide by sizes: $v_a = 0.5$, $v_b = 2.667$.
- $t = -5/\sqrt{3.167} = -2.810$.
- Welch $\nu = 8.04$, so $p = 0.023$.
- Student pooled: $t = -2.445$ on $10$ df, so $p = 0.035$.

The pooled test is more conservative here ($p = 0.035$ against $0.023$), but the direction depends on sizes
and spreads.

## Why it works

Each mean has standard error $s/\sqrt n$, and the difference of two independent means has variance equal to
the sum of the two. Welch keeps the two terms separate, and its df comes from matching the first two moments
of the variance estimate. Student's test assumes $\sigma_a = \sigma_b$, so it pools the variances and uses the
sum of degrees of freedom, which is exact only under that assumption.

Pairing removes variation: the difference $d_i$ cancels the unit effect, while treating the pairs as
independent keeps that variation in the standard error. For eight before/after pairs,
$\text{before} = (10, 12, 9, 14, 11, 13, 8, 15)$ and $\text{after} = (13, 13, 11, 17, 13, 15, 9, 17)$, the
paired test gives $p = 0.00014$, and Welch on the same 16 numbers gives $p = 0.149$.

For ANOVA, under equal means and normal errors, $F$ is a ratio of two independent variance estimates with an
$F(k-1, N-k)$ distribution. With $k=2$ it is the square of the pooled t statistic.

## Signal vs. noise

- **Signal:** Welch by default; paired when units repeat; ANOVA as variance decomposition.
- **Signal:** a significant ANOVA says that at least one mean differs. It does not say which.
- **Noise for now:** the F distribution's derivation.

## Where it breaks

- **Unequal variances with pooling.** Student's test on unequal spreads gives the wrong p-value. For
  $a = (4.4, 5.3, 5.5, 4.3, 5.4, 5.0, 5.0, 4.5)$ and $b = (5.3, 5.4, 8.7)$, the pooled test gives
  $p = 0.049$ (just significant) and Welch gives $p = 0.301$. The variances are $0.22$ and $3.74$. The
  pooled $s_p^2 = 1.00$ is 83% from the 3-value group yet applies to both means, giving a standard error of
  $0.68$ where Welch's is $1.13$.
- **Skew and outliers at small $n$.** Means are sensitive to one extreme value; check small samples before
  trusting a p-value.
- **Dependence.** Repeated measures need the paired or mixed model. Independent-sample tests overstate
  precision.
- **Power.** A non-significant result is not evidence of equal means. With equal spreads, 20 users per group
  and a half-SD difference, the two-sided test at $\alpha = 0.05$ has power $0.34$ (64 per group for $0.80$).
- **Zero spread.** If both groups are constant, the standard error is zero and $t$ is undefined. A
  constant group with a varying other group is fine.
- **Multiple comparisons.** Running many pairwise t-tests after a significant ANOVA inflates the false
  positive rate.
- **Float32 storage.** The setosa and virginica sepal lengths (50 each, unequal spreads) give Welch $t = -15.386196$ in float64. Computed in float32 from the float32 storage, the same statistic gives $t = -15.386199$, a relative change of $2.1 \times 10^{-7}$. ANOVA $F = 119.2645$ on the three species moves by $2.0 \times 10^{-7}$ the same way. That is far above a $10^{-9}$ tolerance, so a solution that never converts to float64 fails on storage alone.

## Interview angle

**60-second answer.** For two independent groups I use Welch's t-test. It does not assume equal variances,
and it loses almost nothing when they are equal. I use Student's pooled test only when equal variance is a
design fact. If the units are paired, I use a paired test on the differences, because pairing removes the
unit-to-unit variation. For more than two groups, one-way ANOVA tests whether any mean differs, by
comparing between-group and within-group variation.

**Follow-ups.**
- *What do you do after a significant ANOVA?* Compare the pairs of interest with a multiple-comparison
  procedure such as Tukey's HSD or Holm, and report an effect size with its interval. Significance alone
  says little about how large the difference is.
- *ANOVA with unequal variances?* Use Welch's version of ANOVA, or check the residuals and the sample sizes
  first.

## Connections

- **Requires:** `cml/sampling-distributions-standard-errors-and-cis` (standard errors and the t distribution).
- **Unlocks:** `cml/a-b-test-design-and-analysis` (two-proportion and mean comparisons in experiments),
  `cml/multiple-testing-corrections` (the cost of many comparisons), `cml/regression-inference-t-f-and-intervals`
  (ANOVA is a regression with indicator variables).
- **Lineage:** Student (1908) introduced the t-test; Welch (1947) gave the unequal-variance version; Fisher
  (1925) developed analysis of variance.
