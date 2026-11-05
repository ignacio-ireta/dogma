# Code Comprehension — What Does This Code Do?

**Format:** Show this code to the candidate. Ask the questions below. Conversational — no coding required.

```python
def detect_anomalies(df, farm_id, window=24, threshold=2.5):
    farm = df[df['farm_id'] == farm_id].copy()
    farm['timestamp'] = pd.to_datetime(farm['timestamp'])
    farm = farm.sort_values('timestamp')

    farm['rolling_mean'] = farm['power_output_mw'].rolling(window, min_periods=1).mean()
    farm['rolling_std'] = farm['power_output_mw'].rolling(window, min_periods=1).std()

    farm['z_score'] = (farm['power_output_mw'] - farm['rolling_mean']) / farm['rolling_std']
    farm['is_anomaly'] = farm['z_score'].abs() > threshold

    return farm[farm['is_anomaly']][['timestamp', 'power_output_mw', 'rolling_mean', 'z_score']]
```

## Questions to Ask

1. **"Walk me through what this function does."**
   - Expected: It detects anomalous power output readings for a specific wind farm using a rolling Z-score approach. It calculates a rolling mean and standard deviation over a 24-period window, computes a Z-score for each observation, and flags any reading more than 2.5 standard deviations from the rolling mean as an anomaly.

2. **"What are the limitations of this approach for wind farm data?"**
   - Expected: Wind power output is naturally variable (depends on wind). A sudden drop in output might just mean the wind stopped — that's not an equipment anomaly. The method doesn't account for wind speed as an input. A better approach would check if output is anomalous GIVEN the current wind conditions, not just relative to recent output.

3. **"The `min_periods=1` parameter — what does it do and why might it cause problems?"**
   - Expected: It allows the rolling window to compute with fewer than `window` observations, which is useful at the start of the series. But with very few periods, the rolling statistics are unreliable — a single observation has a meaningless standard deviation. This could lead to false anomalies at the beginning of the time series.
