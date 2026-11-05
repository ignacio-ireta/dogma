# Coding Problem 2 — Promotion Gap Analysis

**Time:** ~7 minutes

## Setup

Using the same dataset (`employees.csv`).

## Task

Write a Python function that analyzes whether **time since last promotion** is related to attrition, segmented by performance:

1. Bin `last_promotion_months_ago` into: `Recent (0-12)`, `Moderate (13-24)`, `Overdue (25-36)`, `Long Overdue (36+)`
2. Bin `performance_score` into: `High (4-5)`, `Average (3)`, `Low (1-2)`

3. For each **performance_bin** × **promotion_bin**, calculate:
   - `employee_count`
   - `attrition_rate` — Percentage who left (rounded to 1 decimal)
   - `avg_satisfaction` — Average satisfaction score (rounded to 2 decimal). Exclude nulls.
   - `avg_overtime` — Average overtime hours (rounded to 1 decimal)

Sort by `attrition_rate` descending.

**Bonus:** Do high performers with overdue promotions leave at a higher rate than low performers with recent promotions? What does this suggest about the company's retention strategy?
