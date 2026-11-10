---
title: Data quality and drift detection
minutes: 12
min_viable: "Read The system and Where it breaks, then solve p01 (population stability index)."
---

## The system

When a production model degrades, check pipeline quality first. Input drift needs no labels; concept drift needs outcomes that may arrive late.

| Stage | Symptom it explains | Check | Needs labels? | Typical fix |
|---|---|---|---|---|
| 0. Data quality | Errors, empty features, jumps | Schema, nulls, ranges, uniqueness, freshness | No | Fix the pipeline |
| 1. Covariate drift | Input distribution moved, $P(x)$ | PSI or KS per feature, reference against current | No | Recalibrate, or retrain on recent data |
| 2. Label drift | Class mix moved, $P(y)$ | Label shares against training | Yes | Adjust priors or thresholds |
| 3. Concept drift | Same inputs, different labels, $P(y\mid x)$ | Error by segment over time | Yes | Retrain with recent labels |

Failures can coexist. Investigate pipeline errors before treating an alert as model drift.

## Worked example

**PSI.** Reference $(1,2,3,4)$, two bins. The edge is the reference median, $2.5$. Reference shares are $\tfrac12,\tfrac12$. Current $(1,2,3,4,4,4)$ has shares $\tfrac26,\tfrac46$. So $\mathrm{PSI}=(\tfrac13-\tfrac12)\ln\tfrac{1/3}{1/2}+(\tfrac23-\tfrac12)\ln\tfrac{2/3}{1/2}=\tfrac16\ln 2\approx 0.116$. An empty bin takes $\varepsilon = 10^{-6}$ instead of zero.

**KS.** Reference $1..10$, current $6..15$. The largest gap between the two empirical CDFs is $0.5$, reached at every value from $5$ to $10$. With $n_e = 10\cdot10/20 = 5$, $\lambda=\sqrt5\cdot 0.5$ and $p\approx 0.164$. Not drift at $\alpha=0.05$.

## Why it works

**Covariate, label and concept drift** differ in which factor moves [book: Quinonero-Candela et al. 2009]. Covariate drift changes $P(x)$ while $P(y\mid x)$ holds. Label drift changes $P(y)$; pure label shift additionally keeps $P(x\mid y)$ fixed. Concept drift changes $P(y\mid x)$, so the same inputs now mean something else. Input drift is visible without labels, but unlabelled monitors cannot establish that $P(y\mid x)$ stayed fixed.

**PSI** compares two binned distributions with one shared set of bins. The bins come from the reference sample. Independent quantile bins hide drift by equalising shares; use shared reference edges. Values beyond the reference range fall into the outer bins, so they are never dropped.

**KS** is the largest vertical gap between two empirical CDFs. Its Kolmogorov null distribution is an asymptotic approximation for continuous data [paper: Massey 1951]. Tied values must be counted together. The [SciPy KS documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html) states the continuous-distribution assumption; our p-value uses the limiting distribution explicitly, rather than its finite-sample default.

**Data validation** is the cheapest monitor: schema, nulls, value ranges, uniqueness and freshness. These checks expose pipeline failures without a distributional test [Polyzotis et al. 2019, Data Validation for Machine Learning](https://proceedings.mlsys.org/paper_files/paper/2019/hash/928f1160e52192e3e0017fb63ab65391-Abstract.html).

## Signal vs. noise

- **Signal:** the effect size. A KS distance of 0.5 or a PSI of 0.9 is a large change however the p-value reads.
- **Noise:** a p-value alone. With large samples the test flags differences that do not matter. For the same observed KS distance $D=0.02$, the limiting p-value changes from about $0.988$ at $n=1{,}000$ per sample to $8.5\times10^{-18}$ at $n=100{,}000$.
- **Noise:** the 0.1 and 0.25 PSI thresholds. They are conventions from credit-risk practice [book: Siddiqi 2006], not laws. For normal populations separated by half a standard deviation, reference-decile bins give PSI $\approx 0.24$ in the large-sample limit.
- **Noise:** one feature out of hundreds. At $\alpha=0.05$, about one feature in twenty fires by chance.

## Where it breaks

- **Reference window.** A reference from a holiday or a promotion makes every normal week look like drift.
- **Seasonality.** Weekly and daily cycles are not drift. Compare with the same weekday or hour.
- **Bins from the current sample.** The PSI then hides the drift it is meant to show.
- **Empty bins.** A zero share gives $\ln 0$. Floor it at $\varepsilon$, and remember that a bin empty in the current sample can dominate the sum.
- **Discrete features.** The KS p-value assumes a continuous distribution. For a few categories, compare shares by category instead.
- **Monitoring with no action.** Each monitor needs a named response, or the team learns to ignore alerts.

In the offline `adult` snapshot, comparing the `workclass == "?"` indicator between the `<=50K` and `>50K` income
cohorts gives shares of 6.36% and 2.48%, and a two-bin PSI of 0.0380. Dropping those marked rows turns both indicators
into all zeros and reports PSI 0.0000, hiding a 2.6-fold difference in share. For this indicator, `?` is a recorded category value.

## Interview angle

**60-second answer, "Accuracy dropped in production. What do you check first?"** I would check the pipeline before the model: schema, null rates, value ranges and freshness against the last good week. If the data are clean, I would compare each input feature with its training reference, using PSI or a KS distance on the effect size, not only the p-value. Labels may be delayed, so I would check label shares and error by segment only when they arrive. The response depends on the stage: fix the pipeline, recalibrate, or retrain on recent labelled data. I would not retrain first, because retraining on corrupted data repeats the bug.

**Follow-ups.** *How do you monitor without labels?* Watch the input distributions and the prediction distribution against the reference. *Why not retrain on a schedule?* It can be right, but it hides a data bug, and it needs labels that may not exist yet.

## Connections

- **Prerequisites:** `cml/descriptive-statistics` (quantiles, medians) and `cml/bootstrap-and-permutation-resampling` (the null distribution of a statistic).
- **Unlocks:** production monitoring of models and feature stores. The stability idea is the same one that makes a rolling statistic useful in `cml/rolling-statistics-stationarity-and-autocorrelation`.
- **Lineage:** Massey (1951) describes the KS goodness-of-fit test. Quinonero-Candela et al. (2009) and Gama et al. (2014) gave the shift taxonomy used here.
