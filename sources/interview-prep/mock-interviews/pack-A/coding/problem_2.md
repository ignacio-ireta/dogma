# Coding Problem 2 — High-Risk Patient Flagging

**Time:** ~7 minutes

## Setup

Using the same datasets from Problem 1 (`patients.csv` and `admissions.csv`).

## Task

Write a Python function that identifies **high-risk patients** based on the following criteria:

A patient is "high-risk" if they meet **any two** of these conditions:
1. They have **3 or more** chronic conditions
2. They have been admitted to the hospital **more than once** (count of admissions > 1)
3. They have had **2 or more** ER visits in the prior 12 months (use the max `er_visits_prior_12m` across their admissions)

Return a DataFrame with columns:
- `patient_id`
- `age`
- `gender`
- `chronic_conditions`
- `total_admissions` — Number of times they were admitted
- `max_er_visits` — Maximum ER visits recorded across admissions
- `risk_flags` — Count of how many of the 3 criteria they meet (2 or 3)

Sort by `risk_flags` descending, then by `total_admissions` descending.

**Bonus:** What percentage of high-risk patients had at least one readmission?
