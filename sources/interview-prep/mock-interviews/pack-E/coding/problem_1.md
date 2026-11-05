# Coding Problem 1 — Leak Risk by Pipe Characteristics

**Time:** ~7 minutes

## Setup

You have a CSV file:
- `data/sensors.csv` — Sensor readings from the pipe network (sensor_id, timestamp, flow_rate_lpm, pressure_bar, temperature_c, district, pipe_age_years, pipe_material, leak_detected)

Note: Some values may be missing.

## Task

Write a Python function that returns a DataFrame with one row per **pipe_material** × **age_group**, containing:

1. `pipe_material`
2. `age_group` — Bin pipe_age_years into: `New (0-10)`, `Mid (11-25)`, `Old (26-40)`, `Critical (40+)`
3. `sensor_count` — Number of unique sensors in this group
4. `total_readings` — Number of readings
5. `leak_rate` — Percentage of readings where leak_detected = 1 (rounded to 2 decimal)
6. `avg_flow_rate` — Average flow rate (rounded to 1 decimal). Exclude nulls.
7. `avg_pressure` — Average pressure (rounded to 2 decimal). Exclude nulls.

Sort by `leak_rate` descending.

**Bonus:** Which combination of material + age has the highest leak rate?
