# Pack A — Coding Solutions (INTERVIEWER ONLY)

## Problem 1

```python
import pandas as pd

def readmission_summary(patients_path, admissions_path):
    patients = pd.read_csv(patients_path)
    admissions = pd.read_csv(admissions_path)

    merged = admissions.merge(patients[['patient_id', 'age', 'insurance']], on='patient_id', how='left')

    result = merged.groupby('department').agg(
        total_admissions=('admission_id', 'count'),
        readmission_rate=('readmitted_30d', lambda x: round(x.mean() * 100, 1)),
        avg_length_of_stay=('length_of_stay', lambda x: round(x.mean(), 1)),
        avg_patient_age=('age', lambda x: round(x.mean(), 1)),
        pct_medicare=('insurance', lambda x: round((x == 'Medicare').mean() * 100, 1)),
    ).reset_index()

    return result.sort_values('readmission_rate', ascending=False)
```

## Problem 2

```python
def high_risk_patients(patients_path, admissions_path):
    patients = pd.read_csv(patients_path)
    admissions = pd.read_csv(admissions_path)

    admit_stats = admissions.groupby('patient_id').agg(
        total_admissions=('admission_id', 'count'),
        max_er_visits=('er_visits_prior_12m', 'max'),
        any_readmission=('readmitted_30d', 'max'),
    ).reset_index()

    combined = patients.merge(admit_stats, on='patient_id', how='inner')

    combined['flag_chronic'] = (combined['chronic_conditions'] >= 3).astype(int)
    combined['flag_admits'] = (combined['total_admissions'] > 1).astype(int)
    combined['flag_er'] = (combined['max_er_visits'] >= 2).astype(int)
    combined['risk_flags'] = combined['flag_chronic'] + combined['flag_admits'] + combined['flag_er']

    high_risk = combined[combined['risk_flags'] >= 2]

    result = high_risk[['patient_id', 'age', 'gender', 'chronic_conditions',
                         'total_admissions', 'max_er_visits', 'risk_flags']]

    return result.sort_values(['risk_flags', 'total_admissions'], ascending=[False, False])

# Bonus: percentage with readmission
# high_risk['any_readmission'].mean() * 100
```
