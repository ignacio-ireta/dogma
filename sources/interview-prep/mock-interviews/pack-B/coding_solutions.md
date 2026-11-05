# Pack B — Coding Solutions (INTERVIEWER ONLY)

## Problem 1

```python
import pandas as pd
import numpy as np

def farm_performance(path):
    df = pd.read_csv(path)

    def calc_mape(group):
        valid = group[(group['power_output_mw'] > 0) & group['power_output_mw'].notna() & group['forecast_mw'].notna()]
        if len(valid) == 0:
            return np.nan
        return round((abs(valid['power_output_mw'] - valid['forecast_mw']) / valid['power_output_mw']).mean() * 100, 2)

    result = df.groupby('farm_id').agg(
        avg_output=('power_output_mw', lambda x: round(x.mean(), 1)),
        avg_forecast=('forecast_mw', lambda x: round(x.mean(), 1)),
        forecast_bias=('power_output_mw', lambda x: round((df.loc[x.index, 'forecast_mw'] - x).mean(), 1)),
    ).reset_index()

    # MAPE per farm
    mape_series = df.groupby('farm_id').apply(calc_mape).reset_index(name='mape')
    result = result.merge(mape_series, on='farm_id')

    # Data completeness
    completeness = df.groupby('farm_id').apply(lambda g: round(g.dropna().shape[0] / g.shape[0] * 100, 1)).reset_index(name='data_completeness')
    result = result.merge(completeness, on='farm_id')

    return result.sort_values('mape', ascending=False).head(10)
```

## Problem 2

```python
def wind_power_analysis(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['wind_speed_ms'])

    bins = [0, 3, 7, 12, 20, float('inf')]
    labels = ['Calm', 'Light', 'Moderate', 'Strong', 'Extreme']
    df['wind_bin'] = pd.cut(df['wind_speed_ms'], bins=bins, labels=labels, right=True)

    df['abs_error'] = abs(df['forecast_mw'] - df['power_output_mw'])
    df['overpredict'] = df['forecast_mw'] > df['power_output_mw']

    result = df.groupby('wind_bin', observed=True).agg(
        count=('power_output_mw', 'count'),
        avg_power=('power_output_mw', lambda x: round(x.mean(), 1)),
        avg_forecast_error=('abs_error', lambda x: round(x.mean(), 1)),
        pct_overpredict=('overpredict', lambda x: round(x.mean() * 100, 1)),
    )

    return result

# Bonus: Worst bin is likely 'Calm' (0-3 m/s) — model predicts output but turbines
# don't spin below cut-in speed. Linear model doesn't capture this non-linearity.
```
