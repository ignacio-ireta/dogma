# Coding Problem 1 — Applicant Risk Segmentation

**Time:** ~7 minutes

## Setup

You have a CSV file:
- `data/applicants.csv` — Loan applicant data (applicant_id, age, gender, region, has_bank_account, monthly_income_est, mobile_topup_freq, utility_payment_score, mobile_money_txns_monthly, phone_model_tier, location_changes_monthly, loan_amount_requested, defaulted)

Note: Some values may be missing.

## Task

Write a Python function that returns a DataFrame with one row per **region** × **has_bank_account**, containing:

1. `region`
2. `has_bank_account` (0 or 1)
3. `applicant_count` — Number of applicants
4. `default_rate` — Percentage who defaulted (rounded to 1 decimal)
5. `avg_income` — Average estimated monthly income (rounded to 0 decimal). Exclude nulls.
6. `avg_loan_requested` — Average loan amount requested (rounded to 0 decimal)
7. `avg_utility_score` — Average utility payment score (rounded to 1 decimal). Exclude nulls.

Sort by `default_rate` descending.

**Bonus:** Which segment has the highest default rate? Does it surprise you?
