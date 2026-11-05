# Pack D — Coding Solutions

## Problem 1
```python
import pandas as pd

def risk_segmentation(path):
    df = pd.read_csv(path)

    result = df.groupby(['region', 'has_bank_account']).agg(
        applicant_count=('applicant_id', 'count'),
        default_rate=('defaulted', lambda x: round(x.mean() * 100, 1)),
        avg_income=('monthly_income_est', lambda x: round(x.dropna().mean(), 0)),
        avg_loan_requested=('loan_amount_requested', lambda x: round(x.mean(), 0)),
        avg_utility_score=('utility_payment_score', lambda x: round(x.dropna().mean(), 1)),
    ).reset_index()

    return result.sort_values('default_rate', ascending=False)
```

## Problem 2
```python
def feature_analysis(path):
    df = pd.read_csv(path)

    features = ['mobile_topup_freq', 'utility_payment_score',
                'mobile_money_txns_monthly', 'location_changes_monthly', 'monthly_income_est']

    rows = []
    for feat in features:
        defaulted = df[df['defaulted'] == 1][feat].dropna().mean()
        non_defaulted = df[df['defaulted'] == 0][feat].dropna().mean()
        pct_diff = round((defaulted - non_defaulted) / non_defaulted * 100, 1)
        rows.append({
            'feature': feat,
            'mean_defaulted': round(defaulted, 2),
            'mean_non_defaulted': round(non_defaulted, 2),
            'pct_difference': pct_diff,
        })

    result = pd.DataFrame(rows)
    result['abs_pct'] = result['pct_difference'].abs()
    return result.sort_values('abs_pct', ascending=False).drop(columns='abs_pct')
```
