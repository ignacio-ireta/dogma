# Pack E — Coding Solutions

## Problem 1
```python
import pandas as pd

def leak_risk(path):
    df = pd.read_csv(path)

    bins = [0, 10, 25, 40, float('inf')]
    labels = ['New (0-10)', 'Mid (11-25)', 'Old (26-40)', 'Critical (40+)']
    df['age_group'] = pd.cut(df['pipe_age_years'], bins=bins, labels=labels, right=True)

    result = df.groupby(['pipe_material', 'age_group'], observed=True).agg(
        sensor_count=('sensor_id', 'nunique'),
        total_readings=('sensor_id', 'count'),
        leak_rate=('leak_detected', lambda x: round(x.mean() * 100, 2)),
        avg_flow_rate=('flow_rate_lpm', lambda x: round(x.dropna().mean(), 1)),
        avg_pressure=('pressure_bar', lambda x: round(x.dropna().mean(), 2)),
    ).reset_index()

    return result.sort_values('leak_rate', ascending=False)
```

## Problem 2
```python
def pressure_drops(path):
    df = pd.read_csv(path)

    sensor_stats = df.groupby('sensor_id').agg(
        district=('district', 'first'),
        pipe_material=('pipe_material', 'first'),
        pipe_age_years=('pipe_age_years', 'first'),
        avg_pressure=('pressure_bar', lambda x: round(x.dropna().mean(), 2)),
        min_pressure=('pressure_bar', lambda x: round(x.dropna().min(), 2)),
        leak_detected_ever=('leak_detected', 'max'),
    ).reset_index()

    sensor_stats['pressure_drop_pct'] = round(
        (sensor_stats['avg_pressure'] - sensor_stats['min_pressure']) / sensor_stats['avg_pressure'] * 100, 1
    )

    significant = sensor_stats[sensor_stats['pressure_drop_pct'] > 15]
    return significant.sort_values('pressure_drop_pct', ascending=False)

# Bonus: significant[significant['leak_detected_ever'] == 1].shape[0] / len(significant) * 100
```
