---
title: "Split conformal prediction"
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 using Hint 1 if needed."
---

## The system

You have a fitted model $\hat f$ that was trained without the calibration data. You also have a calibration set of $n$ labelled pairs $(X_i,Y_i)$ and a new input $X_{n+1}$ whose label $Y_{n+1}$ is unknown. Split conformal prediction turns the model's errors on the calibration set into an interval (regression) or a set (classification) that covers $Y_{n+1}$ with probability at least $1-\alpha$.

1. **Score each calibration point.** Regression uses $s_i=|Y_i-\hat f(X_i)|$. Classification uses $s_i=1-\hat\pi_i(Y_i)$, where $\hat\pi_i(k)$ is the predicted probability of class $k$.
2. **Pick the threshold.** Compute the rank $k=\lceil (n+1)(1-\alpha)\rceil$ and let $\hat q$ be the $k$-th smallest score. If $k>n$, set $\hat q=+\infty$.
3. **Output.** Regression returns $[\hat f(x)-\hat q,\ \hat f(x)+\hat q]$. Classification returns every class $j$ with $1-\hat\pi(x)_j\le\hat q$.

Symbols: $n$ is the calibration size and $\alpha\in(0,1)$ the miscoverage level, so $1-\alpha$ is the target coverage. Assume exchangeability: the $n+1$ pairs (calibration and test) are exchangeable. I.i.d. draws satisfy this, but exchangeability is weaker: draws without replacement from a finite population are exchangeable without being independent. The model must be fixed before calibration.

## Worked example

**Regression.** Nine calibration residuals are $0.1,0.2,\dots,0.9$. Take $\alpha=0.2$. Then $(n+1)(1-\alpha)=10\cdot0.8=8$, so $k=8$ and $\hat q=0.8$. For a test prediction $\hat f(x)=5.0$, the interval is $[4.2,\,5.8]$.

Now take $\alpha=0.05$. Then $(n+1)(1-\alpha)=9.5$, so $k=10>9$ and the interval is $(-\infty,+\infty)$. Nine points cannot support a finite 95% interval.

**Classification.** Four calibration points have true-class probabilities $0.7,0.6,0.5,0.5$, so their scores are $0.3,0.4,0.5,0.5$. With $\alpha=0.2$, $k=\lceil 5\cdot0.8\rceil=4$ and $\hat q=0.5$. A test row $\hat\pi=(0.6,0.3,0.1)$ has scores $0.4,0.7,0.9$, so the set is $\{0\}$. A row $\hat\pi=(0.45,0.45,0.1)$ has scores $0.55,0.55,0.9$, so the set is empty. An empty set is allowed: it is the honest output when the model is unsure.

Compute the rank exactly, reading $\alpha$ as the decimal you wrote (`Fraction(repr(alpha))`, so $0.7$ is $7/10$). For $n=9$ and $\alpha=0.7$, $(n+1)(1-\alpha)=3$ exactly, but in floats $10\times(1-0.7)=3.0000000000000004$, whose ceiling is 4.

## Why it works

Let $S_1,\dots,S_{n+1}$ be the scores of the calibration points and the test point, with the model fixed. Exchangeability makes every ordering of these $n+1$ values equally likely. Without ties, the test score's rank among them is uniform on $\{1,\dots,n+1\}$. The test score is at most the $k$-th smallest calibration score exactly when its rank is at most $k$. So

$$P\big(S_{n+1}\le S_{(k)}\big)=\frac{k}{n+1}\ \ge\ 1-\alpha\quad\text{for}\quad k=\lceil (n+1)(1-\alpha)\rceil,$$

where $S_{(k)}$ is the $k$-th smallest calibration score. The event $S_{n+1}\le\hat q$ is exactly the event that $Y_{n+1}$ lands in the interval or the set. The "+1" appears because the test point is one of the $n+1$ exchangeable values. Without it, the rank is one short. For continuous scores, the coverage is also at most $1-\alpha+1/(n+1)$, so the guarantee is nearly tight.

This is **marginal** coverage: the probability averages over the calibration set and the test point together. It does not hold for each input $x$. That is the price of making no distributional assumption.

## Signal vs. noise

Signal: the rank formula, the score definitions, the direction of the guarantee (at least $1-\alpha$, marginal), and that the width comes from the score distribution, not the model class. The threshold depends on $n$ only through $k$. For large $n$, $\hat q$ approaches the population $(1-\alpha)$ quantile of the scores, so extra points add little. Noise: variants such as full conformal.

## Where it breaks

- **Exchangeability fails.** Drift, time order or a shifted population breaks the guarantee. Calibrating on last year's data can leave coverage well below $1-\alpha$. Recent calibration narrows the gap but does not restore exchangeability; for time-ordered data, also apply the split rules in [cml/leakage-and-walk-forward-validation].
- **Covariate shift.** If test inputs shift while $P(Y\mid X)$ stays fixed, the calibration scores no longer describe the test scores. Example: $x\sim\mathrm{Uniform}(0,1)$, a fixed predictor $\hat f(x)=x$, and $Y=x+(0.2+2x)\varepsilon$ with standard normal $\varepsilon$. Calibrate on $n=200$ points with $x\in(0,0.5)$ at $\alpha=0.1$, so $k=181$ and the expected coverage on that region is $181/201\approx0.90$. Averaged over 20,000 calibration sets, coverage is 0.90 for test points in $(0,0.5)$ and 0.55 for $x\in(0.5,1)$. This limits the guarantee; it is not a fix, and a shifted test needs a stated model of the shift.
- **The calibration set was used for training.** Training residuals are smaller than residuals on new data, so $\hat q$ is too small and the intervals undercover. Hold the calibration points out.
- **Marginal is not conditional.** Overall coverage can be 90% while one subgroup receives 60%. Report coverage per group. *The Limits of Distribution-Free Conditional Predictive Inference* (2021, per the Library note) shows that distribution-free conditional coverage is generally unattainable without restricting the groups or the data.
- **Small $n$.** When $\alpha<1/(n+1)$ the rank exceeds $n$, and no finite interval exists.
- **Constant width under heteroscedastic noise.** Absolute residuals give one width everywhere: too wide where the model is precise, too narrow where it is uncertain.

## Interview angle

Hold out $n$ points the model never saw and score them: the absolute residual for regression, one minus the true-class probability for classification. The $\lceil (n+1)(1-\alpha)\rceil$-th smallest score is the threshold; output the prediction plus or minus it, or every class under it. Under exchangeability, the new label is covered with probability at least $1-\alpha$, averaged over calibration sets and test points, with no distributional assumption and no change to the model. The guarantee is marginal and breaks under drift, so monitor coverage on labels as they arrive.

Likely follow-ups: why $n+1$ (the test point is exchangeable with the calibration points), and what happens when $\alpha$ is too small for $n$ (an infinite threshold).

## Connections

Requires [cml/sampling-distributions-standard-errors-and-cis] (quantiles of a sampling distribution) and [cml/regression-metrics-and-robust-losses] (absolute residuals). General treatments are in the topic sources; the Library note gives locators for the rank and conditional-coverage results.
