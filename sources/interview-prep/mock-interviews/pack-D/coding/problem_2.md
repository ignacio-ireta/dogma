# Coding Problem 2 — Alternative Data Feature Analysis

**Time:** ~7 minutes

## Setup

Using the same dataset (`applicants.csv`).

## Task

Write a Python function that analyzes which **alternative data features** are most different between applicants who defaulted and those who didn't:

1. For each of these numeric features: `mobile_topup_freq`, `utility_payment_score`, `mobile_money_txns_monthly`, `location_changes_monthly`, `monthly_income_est`

2. Calculate:
   - `mean_defaulted` — Mean value for applicants who defaulted (rounded to 2 decimal)
   - `mean_non_defaulted` — Mean value for those who didn't default (rounded to 2 decimal)
   - `pct_difference` — Percentage difference: `(mean_defaulted - mean_non_defaulted) / mean_non_defaulted * 100` (rounded to 1 decimal)

3. Drop missing values per feature when computing means.

Return a DataFrame with columns: `feature`, `mean_defaulted`, `mean_non_defaulted`, `pct_difference`. Sort by absolute value of `pct_difference` descending.

**Bonus:** Based on the results, which 2 features would you prioritize for a credit scoring model? Why?
