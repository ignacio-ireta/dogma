# Coding Problem 1 — Patient Readmission Summary

**Time:** ~7 minutes

## Setup

You have two CSV files:
- `data/patients.csv` — Patient demographics (patient_id, age, gender, insurance, zip_code, chronic_conditions)
- `data/admissions.csv` — Hospital admissions (admission_id, patient_id, admission_date, department, length_of_stay, discharge_disposition, er_visits_prior_12m, readmitted_30d)

## Task

Write a Python function that takes the file paths as input and returns a DataFrame with one row per **department**, containing:

1. `department` — Department name
2. `total_admissions` — Total number of admissions
3. `readmission_rate` — Percentage of admissions that resulted in 30-day readmission (rounded to 1 decimal)
4. `avg_length_of_stay` — Average length of stay in days (rounded to 1 decimal)
5. `avg_patient_age` — Average age of patients admitted to that department (rounded to 1 decimal)
6. `pct_medicare` — Percentage of admissions where the patient has Medicare insurance (rounded to 1 decimal)

Sort by `readmission_rate` descending.

**Hint:** You'll need to merge the two datasets on `patient_id`.
