# Coding Problem 1 — Wind Farm Performance Summary

**Time:** ~7 minutes

## Setup

You have a CSV file:
- `data/wind_production.csv` — Daily production data across 50 wind farms (timestamp, farm_id, wind_speed_ms, temperature_c, humidity_pct, pressure_hpa, power_output_mw, forecast_mw)

Note: Some values may be missing.

## Task

Write a Python function that takes the file path and returns a DataFrame with one row per **farm**, containing:

1. `farm_id` — Farm identifier
2. `avg_output` — Average daily power output in MW (rounded to 1 decimal)
3. `avg_forecast` — Average daily forecast in MW (rounded to 1 decimal)
4. `mape` — Mean Absolute Percentage Error between forecast and actual output (rounded to 2 decimal). Formula: `mean(|actual - forecast| / actual) * 100`. Exclude rows where actual output is 0 (to avoid division by zero).
5. `forecast_bias` — Average of (forecast - actual). Positive = over-predicting, negative = under-predicting. (rounded to 1 decimal)
6. `data_completeness` — Percentage of rows with no missing values across all columns (rounded to 1 decimal)

Sort by `mape` descending (worst-performing farms first).

Show only the **top 10** worst-performing farms.
