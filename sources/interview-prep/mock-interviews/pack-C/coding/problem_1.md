# Coding Problem 1 — Delivery Performance Analysis

**Time:** ~7 minutes

## Setup

You have two CSV files:
- `data/orders.csv` — Order data (order_id, timestamp, hour, district, order_value_mxn, items, is_premium_customer, delivery_time_min, weather)
- `data/riders.csv` — Rider data (rider_id, home_district, avg_deliveries_per_day, avg_rating, vehicle_type, active_hours_per_week, months_active)

Note: Some values may be missing.

## Task

Write a Python function that returns a DataFrame with one row per **district**, containing:

1. `district` — District name
2. `total_orders` — Total number of orders
3. `avg_delivery_time` — Average delivery time in minutes (rounded to 1 decimal). Exclude nulls.
4. `pct_delayed` — Percentage of orders with delivery time > 45 minutes (rounded to 1 decimal)
5. `avg_order_value` — Average order value in MXN (rounded to 2 decimal). Exclude nulls.
6. `rider_count` — Number of riders whose home district matches this district (from riders.csv)
7. `orders_per_rider` — `total_orders` / `rider_count` (rounded to 1 decimal)

Sort by `pct_delayed` descending.

**Hint:** You'll need to aggregate orders and riders separately, then merge.
