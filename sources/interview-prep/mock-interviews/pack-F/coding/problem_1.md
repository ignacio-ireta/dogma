# Coding Problem 1 — Attrition Analysis by Department

**Time:** ~7 minutes

## Setup

You have a CSV file:
- `data/employees.csv` — Employee data (employee_id, department, tenure_months, performance_score, salary_usd, last_promotion_months_ago, overtime_hours_monthly, satisfaction_score, num_projects, manager_rating, remote_pct, stock_options, left_company)

Note: Some values may be missing.

## Task

Write a Python function that returns a DataFrame with one row per **department**, containing:

1. `department`
2. `headcount` — Total employees
3. `attrition_rate` — Percentage who left (rounded to 1 decimal)
4. `avg_tenure_months` — Average tenure (rounded to 1 decimal)
5. `avg_performance` — Average performance score (rounded to 2 decimal)
6. `avg_satisfaction` — Average satisfaction score (rounded to 2 decimal). Exclude nulls.
7. `avg_overtime` — Average monthly overtime hours (rounded to 1 decimal)
8. `top_performer_attrition` — Attrition rate specifically for employees with performance_score >= 4 (rounded to 1 decimal)

Sort by `top_performer_attrition` descending.

**Bonus:** Which department has the biggest gap between overall attrition and top performer attrition?
