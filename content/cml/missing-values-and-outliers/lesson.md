---
title: Missing values and outliers
minutes: 12
min_viable: "Read The system and Where it breaks, then solve p01."
---

## The system

Two questions come before any fill or any removal: **why is the value missing**, and **is an extreme value an
error or a real observation**. The answers decide the treatment.

| Mechanism | Plain meaning | What it allows |
|---|---|---|
| MCAR | missingness is unrelated to any value, observed or not | complete-case rows remain a random sample; single imputation still distorts uncertainty |
| MAR | conditional on observed variables, missingness is independent of missing values | a correctly specified conditional model can support imputation |
| MNAR | missingness still depends on unobserved values after conditioning on observed variables | identification needs extra assumptions or information |

Three treatments follow from that.

1. **Impute** a numeric gap with a statistic of the observed values: the median or the mean.
2. **Indicate** the gap with a column that is $1$ where the value was missing. The indicator keeps the fact that
   the value was absent, which can be predictive even when the fill is good.
3. **Delete** the rows only when the gap is rare and the mechanism is MCAR.

Outliers use a separate rule. The modified z-score is $M = 0.6745\,(x-\tilde x)/\mathrm{MAD}$, with median
$\tilde x$ and $\mathrm{MAD}=\operatorname{median}(|x-\tilde x|)$. Iglewicz and Hoaglin suggest flagging
$|M|>3.5$; see the [NIST outlier reference](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm). If $\mathrm{MAD}=0$, the score is undefined. This exercise chooses to flag values that differ from the
median; that fallback is a convention, not a universal outlier theorem.

## Worked example

Take $x=(1,2,3,4,100)$. The median is $3$. The deviations are $(2,1,0,1,97)$, so $\mathrm{MAD}=1$.
The modified scores are $0.6745\,(x-3)=(-1.35,-0.67,0,0.67,65.4)$. Only $100$ exceeds $3.5$.

The classic z-score gives a different answer. The mean is $22$ and the population standard deviation (denominator $n$) is about $39$,
so $z_{100}\approx2.0$, below the usual cutoff of $3$. The outlier inflates the very statistics used to
judge it. That is the reason for the median-based rule.

## Why it works

The median and the MAD have breakdown points of one half: roughly half the observations must be contaminated to drive the median without bound.
The exact finite-sample fraction depends on the definition and sample size; MAD can also collapse to zero. The mean and the standard deviation break at a single
value. That is why the modified score can see an outlier that the classic one hides.

Imputation is a claim about the missing values, and the claim is only as good as the mechanism. MAR alone does not justify a
constant median fill or make it statistically consistent for every estimand. A suitable conditional model
and assumptions are needed. Single-value fills shrink variation and do not propagate missing-value uncertainty.
Under MNAR, observed data alone generally cannot identify the missing-value distribution; use sensitivity
analysis or external information. The indicator preserves missingness but does not identify its mechanism.
These distinctions follow [book: Little & Rubin, Statistical Analysis with Missing Data, missing-data mechanisms].

## Signal vs. noise

- **Signal:** fit the fill on training rows only, and keep the indicator. Flag outliers with the rule you
  stated, not with a visual impression.
- **Signal:** a value that is an outlier is not thereby wrong. Check whether it is a data error before
  removing it.
- **Noise for now:** multiple imputation and model-based outlier detection. They refine the same two
  decisions.

## Where it breaks

- **Imputing before splitting.** The median of the full table has seen the test rows. Fit the fill inside
  each training fold, exactly as with a scaler.
- **Zero MAD.** When more than half of the values are equal, the MAD is zero and the score divides by zero.
  The rule must say what happens, and it must not silently return no outliers.
- **Mean-centred scores on skewed data.** On real `engel` income (235 households, skew 2.8), the median and MAD rule flags 7 households at threshold 3.5. Centring on the mean (982.5, against a median of 884.0) flags 6: the upper tail pulls the mean and hides one household. Centre on the median, and check skew before reading flags as errors.
- **Trimming real extremes.** Capping or deleting a true tail changes the target you are modelling. Transform
  or keep it when the extreme is a real observation.
- **Deletion under MNAR.** Dropping rows with a gap keeps only the rows whose missingness the mechanism
  allowed. Under MNAR the remaining sample is biased.

## Interview angle

Start with the mechanism: MCAR, MAR or MNAR, and what evidence you have. Then say what you do with the gap:
an indicator plus a fill fitted on training data, or deletion only when it is rare and random. For outliers,
say which rule you use and why a median-based score is robust to the outlier it is looking for. Follow-ups:
how you would investigate the mechanism (compare missing and observed rows on other variables,
while recognizing that observed data alone cannot establish MAR rather than MNAR), and when you
would keep an extreme value.

## Connections

- **Prerequisites:** `cml/descriptive-statistics` (median, quartiles and the IQR fences), and
  `cml/shuffling-splitting-and-batching-data` (the training rows that a fill may use).
- **Unlocks:** `cml/preprocessing-pipelines` (imputers inside a pipeline), `cml/k-fold-cross-validation`
  (fits per fold), and `cml/regression-metrics-and-robust-losses` (outliers in the loss).
