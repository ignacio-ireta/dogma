# Seasonal naive forecast and MASE

Implement `seasonal_naive_forecast_and_mase(train, test, season)`. The seasonal naive method forecasts each future
value with the value one season earlier, so it repeats the last complete season. It is the baseline every
forecasting model has to beat. The function returns that forecast for the `len(test)` steps after `train`, and
the MASE (mean absolute scaled error) of the forecast against the realised `test` values. MASE divides the mean
absolute error by the in-sample mean absolute error of the same naive rule on the training data. A value of 1 means
the forecast is about as accurate as the naive rule was in-sample.

## Contract

- `train`: 1-D sequence of finite real numbers, length n >= season + 1.
- `test`: 1-D sequence of finite real numbers, length H >= 1. The realised values for the forecast steps. They are
  used only for the error, never for the forecast or the scale.
- `season`: integer >= 1. A bool or a float is rejected. `season = 1` is the naive method, which repeats the last value.
- Forecast for step h = 1 .. H, with train positions counted from 1 and n = len(train):

      yhat(n + h) = y(n + h - season * ceil(h / season))

- Scale: s = mean of |y_t - y_(t - season)| over t = season + 1 .. n. This has n - season terms.
- MASE = (mean over h of |test_h - yhat(n + h)|) / s.
- Returns: `(forecast, mase)`, where `forecast` is a float64 array of length H and `mase` is a float.
- Raises `ValueError` when an input is not 1-D or has NaN or infinity; when `test` is empty; when `season` is not a
  positive integer; when `len(train) < season + 1`; or when the scale is zero, because the training series repeats
  exactly at lag `season` and MASE is then undefined.
- Must not mutate its inputs.

## Examples

```python
>>> seasonal_naive_forecast_and_mase([10, 22, 31, 12, 19, 33, 9, 21], [20, 25, 11, 30], 3)
(array([33.,  9., 21., 33.]), 4.375)
```

```python
>>> seasonal_naive_forecast_and_mase([1, 2, 3, 4], [5, 6], 1)
(array([4., 4.]), 1.5)
```

In the first example the last complete season is 33, 9, 21, so the forecast is 33, 9, 21, 33. The test errors
are 13, 16, 10 and 3, with mean 42 / 4 = 10.5. The in-sample differences at lag 3 are 2, 3, 2, 3 and 2, with mean
12 / 5 = 2.4, so MASE = 10.5 / 2.4 = 4.375. Dividing by n = 8 instead of 5 would give 7.0.

## Constraints

- Allowed: NumPy and the Python standard library.
- Not allowed: `statsmodels` and `pandas`. Their forecasting and accuracy helpers are the thing this problem asks you
  to write, so the naive forecast and the scale must be built from the formulas above.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those compare the forecast and the MASE with an
explicit, loop-based implementation of the formulas, on seeded random series with several seasons, including a
horizon longer than one season. They check that a season of one gives the naive method, that a horizon longer than
one season repeats the last complete season, and that the scale is the mean over n - season differences. They also
check invariance under shifting and scaling train and test together, that the forecast depends only on the training
data, and that the scale does not depend on the test values. The tests cover a zero-scale training series, a training
series too short for its season, invalid `season`, an empty `test`, non-finite values, a two-dimensional input,
integer input, and inputs that must not be mutated.
