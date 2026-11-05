# Interview notes: Rolling statistics, stationarity and autocorrelation

## Q1. How would you forecast demand for this product?

**60-second answer.** I would start with the seasonal naive forecast, evaluated on walk-forward origins, because
it sets the bar any model has to clear. I would plot the series, then check whether its level and spread are stable
with rolling statistics. I would difference only when the data need it, once for trend and once at the season length
if there is a repeating cycle. Then I would read the ACF and PACF to propose two or three ARIMA orders, and the
holdout would decide between them. I would score each candidate with MASE against the seasonal naive scale, because
it is scale-free and it does not break at zero demand. I would report errors by horizon, and I would monitor residuals
after launch, because a regime shift is the most likely failure.

**Follow-ups.**
- *What would make you skip ARIMA entirely?* A baseline that is already within the noise of the best candidate on walk-forward origins.
- *How do you handle holidays?* Add them as known future regressors, since a fixed season length cannot place them.

## Q2. How do you choose the orders of an ARIMA model?

**60-second answer.** The order of the model is a family decision, not a search. First choose $d$, the number of
differences, from stationarity checks and the rolling statistics. Then read the ACF and PACF of the differenced series.
A cut-off in the PACF after lag $p$ suggests AR($p$), a cut-off in the ACF after lag $q$ suggests MA($q$), and two
tails suggest a small ARMA model. I would fit two or three candidates, compare them on held-out origins and with an
information criterion, and check that the residuals look like white noise. The plots propose orders; they rarely
settle them on real data.

**Follow-ups.**
- *What does a lag-1 ACF near -0.5 after differencing mean?* It can suggest over-differencing; check the undifferenced series and residuals.
- *Why not pick the order that minimises in-sample error?* Extra terms can improve fit while worsening forecasts; evaluate held-out origins.

## Q3. Why MASE rather than MAPE or RMSE?

**60-second answer.** RMSE is in the units of the series, so it compares models only on the same series. MAPE is
scale-free, but it divides by the actual value, so it is undefined at zero, and it penalises over-forecasts more than
under-forecasts of the same ratio. MASE divides the mean absolute test error by the mean absolute error of the seasonal
naive rule on the training data. A value below one beats the training seasonal-naive MAE; it does not establish a win over the baseline on the test period. It works at zero actual values when its training scale is nonzero.

**Follow-ups.**
- *What does MASE equal for the seasonal naive forecast itself?* It need not be 1: the denominator is training one-step seasonal error, while test horizons and regimes can differ.
- *When would you still report RMSE?* When the loss the business pays is quadratic, such as large misses being costly beyond their size.

## Common mistakes

- Starting with ARIMA and skipping the naive and seasonal naive baselines.
- Using a random k-fold split on a time series, which trains on the future.
- Reading the ACF on a non-stationary series and concluding that the orders are large.
- Using a centred rolling window, or using the current observation before it is available at prediction time.
- Reporting MAPE for a series with zeros.
