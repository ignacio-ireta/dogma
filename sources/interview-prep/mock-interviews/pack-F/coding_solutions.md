# Pack F — Coding Solutions

## Problem 1
```python
import pandas as pd

def attrition_by_dept(path):
    df = pd.read_csv(path)

    overall = df.groupby('department').agg(
        headcount=('employee_id', 'count'),
        attrition_rate=('left_company', lambda x: round(x.mean() * 100, 1)),
        avg_tenure_months=('tenure_months', lambda x: round(x.mean(), 1)),
        avg_performance=('performance_score', lambda x: round(x.mean(), 2)),
        avg_satisfaction=('satisfaction_score', lambda x: round(x.dropna().mean(), 2)),
        avg_overtime=('overtime_hours_monthly', lambda x: round(x.mean(), 1)),
    ).reset_index()

    top_perf = df[df['performance_score'] >= 4].groupby('department').agg(
        top_performer_attrition=('left_company', lambda x: round(x.mean() * 100, 1))
    ).reset_index()

    result = overall.merge(top_perf, on='department', how='left')
    return result.sort_values('top_performer_attrition', ascending=False)

# Bonus: result['gap'] = result['top_performer_attrition'] - result['attrition_rate']
```

## Problem 2
```python
def promotion_gap_analysis(path):
    df = pd.read_csv(path)

    promo_bins = [0, 12, 24, 36, float('inf')]
    promo_labels = ['Recent (0-12)', 'Moderate (13-24)', 'Overdue (25-36)', 'Long Overdue (36+)']
    df['promotion_bin'] = pd.cut(df['last_promotion_months_ago'], bins=promo_bins, labels=promo_labels, right=True)

    def perf_bin(score):
        if score >= 4: return 'High (4-5)'
        elif score == 3: return 'Average (3)'
        else: return 'Low (1-2)'
    df['performance_bin'] = df['performance_score'].apply(perf_bin)

    result = df.groupby(['performance_bin', 'promotion_bin'], observed=True).agg(
        employee_count=('employee_id', 'count'),
        attrition_rate=('left_company', lambda x: round(x.mean() * 100, 1)),
        avg_satisfaction=('satisfaction_score', lambda x: round(x.dropna().mean(), 2)),
        avg_overtime=('overtime_hours_monthly', lambda x: round(x.mean(), 1)),
    ).reset_index()

    return result.sort_values('attrition_rate', ascending=False)
```
