---
title: Rolling statistics, stationarity and autocorrelation
minutes: 12
min_viable: "Read The system and Interview angle, then solve p01 (rolling mean and standard deviation)."
---

## The system

Forecasting follows an ordered procedure: establish a baseline, diagnose the series, then compare forecasts. Notation: $y_1,\dots,y_n$ is the series, $h$ the horizon, $m$ the season length.

| Step | Question | Output |
|---|---|---|
| 1. Plot | Trend, season, break, outlier? | Features |
| 2. Baselines | What does the simplest rule achieve? | Naive $y_n$; seasonal naive $y_{n+h-m\lceil h/m\rceil}$ |
| 3. Stationarity | Do mean and spread stay put? | Rolling mean and std; ADF and KPSS tests |
| 4. Difference | Remove trend $\nabla y_t=y_t-y_{t-1}$ or season $\nabla_m y_t=y_t-y_{t-m}$ | Differenced series |
| 5. ACF and PACF | Which AR and MA orders are plausible? | Two or three $(p,q)$ |
| 6. Compare | Does a candidate beat the baseline on held-out data? | Scale-free error |
| 7. Walk-forward | Does it hold across time? | Errors by origin and horizon |

Steps 2 and 7 are never skipped.

**Decision rules.** Difference once if the rolling mean or spread drifts or ADF fails to reject; difference at lag $m$ for a spike there; then test again. Cross-check with KPSS, whose null is stationarity. ADF rejects and KPSS does not: stationary. ADF fails and KPSS rejects: unit root. Both reject, or neither does: conflicting evidence or too little data, so inspect the plot. Stop at the first difference that passes.

For p01, keep count $c$, mean $a$, and squared-deviation sum $M_2$ for the trailing window of width $w$.
After adding $z$, increment $c$, set $δ=z-a$, $a'=a+δ/c$, then $M_2'=M_2+δ(z-a')$.
To remove $v$ from a count $c>1$, use $a'=a+(a-v)/(c-1)$ and
$M_2'=M_2-(v-a')(v-a)$, then decrement $c$. The sample spread is $\sqrt{M_2/(c-1)}$.
Centre near an observed value and rescale before squaring at extreme magnitudes, scaling the mean and $M_2$ together.

## Worked example

**Rolling window.** For $x=(1,2,3,4,5)$ with trailing window $w=3$, position $i$ uses $x_{i-2},x_{i-1},x_i$. The means are $(\text{NaN},\text{NaN},2,3,4)$ and the sample standard deviations are $(\text{NaN},\text{NaN},1,1,1)$.

**Autocorrelation.** For $x=(1,3,2,5,4,6)$, the mean is $3.5$ and $\sum d_t^2 = 17.5$ for $d_t=x_t-3.5$. Lag 1 sums five products to $1.75$, so $r_1 = 0.10$. Lag 2 sums four products to $6.00$, so $r_2 = 0.343$.

**MASE.** Train $(10,22,31,12,19,33,9,21)$, $m=3$, test $(20,25,11,30)$. The seasonal naive forecast is $(33,9,21,33)$, with test errors $13,16,10,3$ (mean $10.5$). In-sample differences $|y_t-y_{t-3}|$ are $2,3,2,3,2$ (mean $2.4$). So MASE $=10.5/2.4=4.375$.

## Why it works

**Weak stationarity** requires constant finite mean and variance, and covariance that depends only on lag. The Dickey–Fuller test [paper: Dickey & Fuller 1979] has a unit-root null. The augmented version (ADF) adds lagged differences and keeps that null under a chosen deterministic specification. Rejection does not rule out changing variance or structural breaks. See [Hyndman and Athanasopoulos, ARIMA models](https://otexts.com/fpp3/arima.html).

**Differencing** removes a locally linear trend. It can overshoot: for white noise the first difference has lag-1 autocorrelation $-\tfrac12$, because $\operatorname{Cov}(\nabla w_t,\nabla w_{t-1})=-\sigma^2$ and $\operatorname{Var}(\nabla w_t)=2\sigma^2$.

**ACF and PACF** identify the family [book: Box et al., ARMA identification]:

| Pattern | Suggests |
|---|---|
| ACF tails off; PACF cuts off after $p$ | AR($p$) |
| ACF cuts off after $q$; PACF tails off | MA($q$) |
| Both tail off | ARMA($p,q$): compare small orders by AIC or BIC |
| ACF near 1, decaying slowly | Possible unit root or persistent stationary process: investigate |
| Spike at lag $m$ | Investigate seasonality; difference only if needed |

**ARIMA($p,d,q$)** uses $d$ differences and ARMA orders $p,q$. Confirm plot-based candidates with information criteria and residual checks.

**Metrics.** MAE and RMSE carry the series' units. MAPE divides by the actual: at actual 100, forecasts 200 and 50 score 100% and 50%; at zero it is undefined. MASE divides the test MAE by the in-sample seasonal naive MAE [paper: Hyndman & Koehler 2006]. Below 1 beats the **training** seasonal-naive MAE; compare models with the baseline on the same held-out origins to establish a forecast gain. The scale must be nonzero [Hyndman and Athanasopoulos, scaled errors](https://otexts.com/fpp3/toolbox.html#scaled-errors).

**Walk-forward evaluation** fits through $t$, forecasts $h$ steps, scores, then advances the origin. Random folds can train on future values.

## Signal vs. noise

- **Signal:** beat seasonal naive on held-out origins before adding complexity.
- **Signal:** diagnose stationarity before interpreting the ACF or fitting.
- **Noise:** short-series ACF spikes barely outside the approximate white-noise bands, $\pm 1.96/\sqrt{n}$.
- **Noise:** small AIC gaps. Let held-out origins decide.

## Where it breaks

- **Regime shifts.** A coefficient learned before a break keeps predicting the old level. Refit or shorten the training window.
- **Variance that grows with the level.** Consider logs for strictly positive values, or an appropriate variance-stabilising transform.
- **Seasonality that moves.** Holidays shift, and a fixed $m$ misses them.
- **Look-ahead.** Centred windows read future values. A trailing window is causal even if precomputed; shift it if the prediction precedes the current observation.
- **Over-differencing.** A lag-1 ACF near $-0.5$ after differencing can suggest an unnecessary difference; it is not proof.
- **Integer input.** Nile flow volumes are integers, but a 10-year window mean is a fraction. Floor division of the window sum loses up to 0.90 per window (mean 0.44, in 85 of 91 windows). The level drop stays visible (window means 1110.7 for the windows ending 1880-1884, 836.8 for those ending 1920-1924), so the error is silent. Check the fractional part, not the shape.
- **Long lags on real data.** The $n-k$ divisor is invisible at short lags and large at long ones. On sunspots ($n = 309$), the biased estimator gives $r_{10} = 0.659$ and $r_{100} = 0.166$. The $n-k$ divisor gives 0.681 and 0.245, 48% higher at lag 100. Use the biased estimator the contract defines, and say which one you computed.

## Interview angle

**60-second answer, "How would you forecast demand?"** Evaluate seasonal naive on walk-forward origins. Plot level and spread; difference only when needed. Use ACF/PACF to propose ARIMA candidates, compare with the baseline on the same origins, report error by horizon, and monitor residuals.

**Follow-ups.** *Random k-fold?* It can train on future values. *MASE over MAPE?* It permits zero actuals when the training scale is nonzero.

## Connections

- **Prerequisites:** `cml/descriptive-statistics` (mean, ddof) and `cml/covariance-and-correlation` (autocorrelation is correlation with a lagged copy).
- **Unlocks:** forecasting with regressors, and `inf/data-quality-and-drift-detection`, which applies the same stability idea to model inputs.
- **Lineage:** Box and Jenkins popularised ACF and PACF identification. Dickey and Fuller (1979) gave the unit-root test. Hyndman and Koehler (2006) proposed MASE.
