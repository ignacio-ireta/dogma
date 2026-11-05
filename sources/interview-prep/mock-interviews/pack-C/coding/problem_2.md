# Coding Problem 2 — Peak Hour Demand Analysis

**Time:** ~7 minutes

## Setup

Using the same order dataset (`orders.csv`).

## Task

Write a Python function that analyzes how **weather** impacts delivery performance during **peak hours**:

1. Define peak hours as: 12-13 (lunch) and 19-21 (dinner). All other hours are "off-peak."

2. Create a summary DataFrame with one row per **weather condition** × **peak/off-peak**, containing:
   - `weather` — Weather condition
   - `is_peak` — True/False
   - `order_count` — Number of orders
   - `avg_delivery_time` — Average delivery time (rounded to 1 decimal)
   - `pct_delayed` — Percentage of orders > 45 min (rounded to 1 decimal)
   - `avg_order_value` — Average order value (rounded to 2 decimal)

3. Sort by `avg_delivery_time` descending.

**Bonus:** What's the most challenging combination of weather + peak/off-peak for delivery times? Can you quantify how much worse it is compared to the best-case scenario?
