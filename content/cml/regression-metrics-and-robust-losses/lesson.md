---
title: Regression metrics and robust losses
minutes: 12
min_viable: "Read The system and Where it breaks, then solve p01."
---

## The system

Predictions $\hat{\mathbf y}\in\mathbb R^n$ are scored against targets $\mathbf y\in\mathbb R^n$, for $n\ge1$ samples. The scalar residual is $e_i = y_i - \hat y_i$, and $\bar y$ is the sample mean.

| Score | Formula | What it summarises | Units |
|---|---|---|---|
| MAE | $\frac1n\sum_i \lvert e_i\rvert$ | typical absolute miss | units of $y$ |
| MSE | $\frac1n\sum_i e_i^2$ | average squared miss | units of $y$ squared |
| RMSE | $\sqrt{\text{MSE}}$ | MSE in units of $y$ | units of $y$ |
| $R^2$ | $1 - \dfrac{\sum_i e_i^2}{\sum_i (y_i-\bar y)^2}$ | error relative to predicting $\bar y$ | none |

The loss fixes the statistic. For a constant prediction $c$, $\sum_i (y_i-c)^2$ is minimised at the
**mean**, and $\sum_i \lvert y_i-c\rvert$ is minimised at the **median**.

Robust losses cap the derivative with respect to a large scalar residual $r$. The Huber loss, with scalar threshold $\delta>0$, is

$$\ell_\delta(r) = \begin{cases}\tfrac12 r^2 & \lvert r\rvert\le\delta \\ \delta\,(\lvert r\rvert - \tfrac12\delta) & \text{otherwise,}\end{cases}
\qquad \psi_\delta(r) = \ell_\delta'(r) = \begin{cases} r & \lvert r\rvert\le\delta\\ \delta\,\mathrm{sign}(r) & \text{otherwise.}\end{cases}$$

Inside the threshold it behaves like MSE. Outside it, the slope is capped at $\delta$, as in MAE.

## Worked example

Let $\mathbf y = (1, 2, 3, 4, 100)$ and predict a constant $c$ for every row.

1. At $c = 22$ (the mean): MSE $= 1522$ and MAE $= 31.2$.
2. At $c = 3$ (the median): MSE $= 1883$ and MAE $= 20.2$.
3. Each score is lowest at its own centre. The outlier $100$ moves the mean far more than the median.

Now $R^2$. Take $\mathbf y = (1,2,3,4)$ and $\hat{\mathbf y} = (4,3,2,1)$. The residuals are $(-3,-1,1,3)$, so
$\mathrm{SS}_\text{res} = 20$. The deviations of $\mathbf y$ from $\bar y = 2.5$ give $\mathrm{SS}_\text{tot} = 5$.
Therefore $R^2 = 1 - 20/5 = -3$. Predicting the mean would score exactly $0$.

Huber on $r = (0.5, -0.5, 3)$ with $\delta = 1$: the per-residual losses are $(0.125, 0.125, 2.5)$, so the mean
loss is $0.917$. The gradient with respect to each residual is $\psi/n = (0.5, -0.5, 1)/3$. The outlier
contributes $1/3$, where squared error would give $3/3$.

## Why it works

For squared error, $\frac{d}{dc}\sum_i (y_i-c)^2 = -2\sum_i (y_i - c) = 0$ gives $c = \bar y$. Absolute error has
derivative $\sum_i \mathrm{sign}(c - y_i)$, which changes sign at the median.

$R^2$ compares the model with the best constant. Its denominator is the squared error of predicting $\bar y$,
so the mean predictor scores $0$. A model worse than that has a numerator above the denominator, and $R^2$ turns negative.

The Huber residual derivative is bounded by $\delta$. For regression coefficients the contribution is also multiplied by the feature vector, so high-leverage feature outliers remain a risk
(paper: Huber 1964, Robust estimation of a location parameter, Ann. Math. Stat.).

## Signal vs. noise

- **Signal:** MAE is the median's loss and MSE is the mean's loss. $R^2$ is relative to the mean predictor and
  can be negative. The Huber gradient is capped at $\delta$.
- **Signal:** a new model should beat $R^2 = 0$ on held-out data.
- **Noise for now:** the printing of the scores, and the tuning of $\delta$.

## Where it breaks

- **Outliers under MSE.** One bad point can dominate the mean, so RMSE reflects that point, not the bulk.
  Look at the residual distribution before trusting RMSE.
- **$R^2$ has no floor.** Predictions worse than the mean give a value below $0$.
- **$R^2$ depends on target spread.** For $\mathbf y=(1,2,3,4)$ and $\hat{\mathbf y}=\mathbf y+1$, RMSE is $1$ and $R^2=1-4/5=0.2$. For targets $10\mathbf y$ and predictions $10\mathbf y+1$, RMSE stays $1$, but $R^2=1-4/500=0.992$.
- **Nested OLS training $R^2$ never falls when a feature is added.** This assumes both models minimise unpenalised squared error on the same rows. Compare models on held-out data.
- **$R^2$ is undefined for constant $\mathbf y$.** `sklearn`'s `r2_score` returns $1.0$ or $0.0$ there instead of
  raising, so check the convention before comparing libraries.
- **MAPE divides by $|y|$.** A zero target has no defined percentage error. For positive targets, it weights absolute errors by $1/y$, so small actuals dominate and can favour low forecasts.
- **Huber's $\delta$ sets the regime.** Too small and the fit is MAE-like and slow to converge. Too large and it is MSE.

**Real data: squared correlation is not $R^2$.** On the first 6000 California housing rows, the least-squares fit has $R^2 = 0.607$ and $\mathrm{corr}^2 = 0.607$. The two agree only for an in-sample fit with an intercept. Double the predictions and $\mathrm{corr}^2$ stays at $0.607$, while $R^2$ falls to $-2.70$. The target is also censored at 5.00001, so 241 rows tie at the cap. RMSE is $0.73$ against MAE $0.54$: the largest 10% of residuals carry 59% of the squared error.

## Interview angle

*"Which metric would you report, and why?"* Start with the decision the number supports. RMSE penalises big misses
and is in the target's units. MAE is the typical miss, and it is the median's loss. $R^2$ is relative to the mean,
so it is not a portable accuracy figure. Name the trade-off: a robust loss such as Huber bounds an outlier's
influence, at the cost of choosing $\delta$.

Follow-ups:

- *Why does the mean minimise squared error?* Setting $-2\sum_i(y_i-c)$ to zero gives $c=\bar y$.
- *What does a negative $R^2$ mean?* The model is worse than predicting the mean for every row.

Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/normal-equation` (least squares minimises MSE), `cml/descriptive-statistics` (mean and
  median), `cml/gradient-descent` (the Huber gradient drives updates).
- **Unlocks:** `cml/ridge-regression-and-weight-decay` and `cml/lasso-and-elastic-net`, which penalise the same
  squared-error loss. `cml/missing-values-and-outliers` builds on the robustness idea.
- **Lineage:** Huber's 1964 robust location estimator, which joins the squared loss near zero to the absolute loss in the tails.
