# Code Comprehension — What Does This Code Do?

**Format:** Show this code to the candidate. Ask the questions below. This is conversational — no coding required.

```python
def analyze_readmissions(df):
    df['admission_date'] = pd.to_datetime(df['admission_date'])
    df = df.sort_values(['patient_id', 'admission_date'])

    df['prev_discharge'] = df.groupby('patient_id')['admission_date'].shift(1)
    df['days_since_last'] = (df['admission_date'] - df['prev_discharge']).dt.days

    result = df.dropna(subset=['days_since_last'])
    result = result[result['days_since_last'] <= 30]

    dept_counts = result.groupby('department').size().reset_index(name='rapid_readmissions')
    total_counts = df.groupby('department').size().reset_index(name='total')

    merged = dept_counts.merge(total_counts, on='department')
    merged['rapid_rate'] = (merged['rapid_readmissions'] / merged['total'] * 100).round(1)

    return merged.sort_values('rapid_rate', ascending=False)
```

## Questions to Ask

1. **"Walk me through what this code does, line by line."**
   - Expected: It calculates the rate of "rapid readmissions" (within 30 days of a previous admission) by department. It uses shift() to get each patient's previous admission date, computes the gap, filters for gaps ≤ 30 days, and aggregates by department.

2. **"Is there a potential issue with this approach?"**
   - Expected: The shift-based approach only looks at consecutive admissions. If a patient has admissions on Jan 1, Jan 20, and Feb 15, it would flag Jan 20 (19 days after Jan 1) as a rapid readmission, but Feb 15 (26 days after Jan 20) would also be flagged — even though it's 45 days from the original Jan 1 admission. Depending on the business definition, this could over-count readmissions.

3. **"How would you improve this code?"**
   - Could add docstrings and comments
   - Could validate input data types
   - The `dropna` approach is correct but could be more explicit
   - Could add a parameter for the readmission window (not hardcode 30)
