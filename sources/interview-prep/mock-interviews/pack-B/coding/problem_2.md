# Coding Problem 2 — Wind Speed Binning & Power Curve Analysis

**Time:** ~7 minutes

## Setup

Using the same dataset (`wind_production.csv`).

## Task

Write a Python function that analyzes the relationship between wind speed and power output across all farms:

1. **Bin wind speed** into the following categories:
   - `Calm`: 0-3 m/s
   - `Light`: 3-7 m/s
   - `Moderate`: 7-12 m/s
   - `Strong`: 12-20 m/s
   - `Extreme`: >20 m/s

2. For each bin, calculate:
   - `count` — Number of observations
   - `avg_power` — Average power output (rounded to 1 decimal)
   - `avg_forecast_error` — Average absolute error |forecast - actual| (rounded to 1 decimal)
   - `pct_overpredict` — Percentage of observations where forecast > actual (rounded to 1 decimal)

3. Drop rows where `wind_speed_ms` is missing.

Return the result as a DataFrame indexed by wind speed bin (in the order listed above).

**Bonus:** In which wind speed bin does the current forecasting model perform worst? Why do you think that is?
