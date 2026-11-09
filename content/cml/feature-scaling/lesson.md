---
title: Feature scaling
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 using training statistics."
---

## The system

Scaling chooses coordinates for a model. A feature measured in milliseconds should
not automatically outweigh another measured in seconds. Separate **fitting** the
coordinate system from **applying** it: compute statistics from training rows,
then reuse those statistics for every validation, test and production row.

Let training matrix $X$ have shape $(n,d)$: $n$ observations and $d$ features.
For feature $j$, define its mean $\mu_j = n^{-1}\sum_i X_{ij}$ and its population
standard deviation $s_j = \sqrt{n^{-1}\sum_i(X_{ij}-\mu_j)^2}$.
For a new matrix $V$ of shape $(m,d)$, return
$Z_{ij}=(V_{ij}-\mu_j)/s_j$. Means and scales have shape $(d,)$ and broadcast
across rows. A constant training column uses scale 1; it still subtracts its mean.

| Model mechanism | Why units matter |
|---|---|
| Euclidean distance | Squaring a large-unit feature can dominate nearest-neighbor selection |
| Gradient descent | Very different feature scales can create an elongated loss surface |
| L1/L2 penalty | A coefficient's size depends on the unit used for its feature |
| Exact threshold tree | Strictly increasing transformations preserve candidate row partitions |

Choose the transformation to match the mechanism. Standardization is a useful
default for many distance-based and regularized linear models; it is not a
universal prerequisite for learning.

## Worked example

Fit on two rows, $X=[(1,10),(3,14)]$. The mean is $(2,12)$.
The centered rows are $(-1,-2)$ and $(1,2)$; their mean squared deviations
are $(1,4)$, so the scales are $(1,2)$.

An unseen row $(5,8)$ becomes $((5-2)/1,(8-12)/2)=(3,-2)$.
Its coordinates describe displacement from the *training* center. Recomputing a
mean on that single row would produce zeros and erase this information.

For training rows $(4,1)$ and $(4,3)$, means are $(4,2)$ and scales are $(1,1)$
after the constant-column rule. An unseen row $(7,2)$ becomes $(3,0)$:
a constant training feature does not imply that every future value is identical.

## Why it works

A nonconstant training column has transformed mean zero because its centered
values sum to zero. Its population mean squared transformed value is one because
we divide by the square root of that same mean squared deviation. These are
algebraic properties of the training sample, not promises about future samples.

The choice of denominator is a convention: this task uses $n$, not the $n-1$
used in the unbiased sample variance estimator. The
[StandardScaler reference](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
documents this population convention and its constant-feature behavior.
Standardization changes location and scale; it does not make a skewed feature
Gaussian. This distinction matters when diagnosing a model's assumptions.

## Signal vs. noise

- **Signal:** fit after the split, inside each training fold; keep fitted statistics
  with the model. The [scaling documentation](https://sklearn.org/stable/modules/generated/sklearn.preprocessing.scale.html)
  explicitly describes the leakage risk of fitting before splitting.
- **Signal:** scaling features and normalizing each row to unit length solve
  different problems. Row normalization changes relative distances differently.
- **Noise for now:** choosing among many transformations before establishing a
  leakage-free baseline. Compare alternatives using training-only validation.

## Where it breaks

Outliers can move the mean and scale enough to compress typical observations.
Median and IQR scaling is a separate policy, the subject of p02. It needs its own zero-IQR rule. This topic uses the rule that scikit-learn's `RobustScaler` applies to its array scales: an IQR below ten machine epsilons becomes 1.0. Min-max scaling does not guarantee that unseen values stay between zero and one.

On raw `digits` (1797 × 64), 20 pixel columns have IQR zero, and 17 of them still vary. A candidate who special-cases only constant columns keeps scale zero for those 17, so in just the first 60 rows all 1020 of their entries (60 × 17) become infinite or NaN. The zero-IQR rule must cover varying columns too, and the standard deviation is no substitute for the IQR.

Constant columns need an explicit fallback. Tiny nonzero scales are not constant:
replacing every small scale with one changes the requested transformation.
For numerical stability, divide centered values by their largest magnitude
before squaring, then restore that magnitude after the square root.

Missing values need a stated policy before scaling. This problem rejects NaN and
infinity. A production pipeline could impute first, but the imputer must also fit
on training data only. Finite inputs can still overflow. Training values $0$ and $10^{-300}$ have scale
$5\cdot10^{-301}$, so the finite value $10^{100}$ maps to $\infty$.

## Interview angle

Explain the mechanism before saying “always scale.” For a regularized linear
model, unequal units change the meaning of equal coefficient penalties; for a
distance model, they change the metric. For exact threshold trees, row ordering
is what matters. Then describe a training-only pipeline that prevents leakage.
Follow-ups: how to handle constants, and how to preserve serving consistency.

## Connections

The prerequisite split defines which rows may influence fitting. Descriptive
statistics supply means and variances. Preprocessing pipelines package these
choices with an estimator; cross-validation repeats their fitting per fold.
The familiar standardized score becomes a fitted software component here. The regularization
connection is in James et al., *An Introduction to Statistical Learning*, the chapter on linear model
selection and regularization (listed in `topic.yaml`).
