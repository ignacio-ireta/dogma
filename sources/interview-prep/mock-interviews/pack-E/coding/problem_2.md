# Coding Problem 2 — Pressure Drop Analysis

**Time:** ~7 minutes

## Setup

Using the same dataset (`sensors.csv`).

## Task

Write a Python function that detects **significant pressure drops** across the sensor network:

1. For each sensor, calculate the **average pressure** over all its readings (excluding nulls).
2. For each sensor, find the **minimum pressure** recorded.
3. Calculate the **pressure_drop_pct**: `(avg_pressure - min_pressure) / avg_pressure * 100`
4. A sensor has a "significant drop" if `pressure_drop_pct > 15%`.

Return a DataFrame with columns:
- `sensor_id`
- `district`
- `pipe_material`
- `pipe_age_years`
- `avg_pressure` (rounded to 2 decimal)
- `min_pressure` (rounded to 2 decimal)
- `pressure_drop_pct` (rounded to 1 decimal)
- `leak_detected_ever` — 1 if ANY reading for that sensor had a leak, 0 otherwise

Filter to only sensors with significant drops. Sort by `pressure_drop_pct` descending.

**Bonus:** What percentage of sensors with significant pressure drops also had a detected leak? What does this tell you about the relationship?
