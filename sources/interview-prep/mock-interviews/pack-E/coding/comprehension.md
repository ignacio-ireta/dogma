# Code Comprehension — What Does This Code Do?

**Format:** Show this code. Ask the questions below. Conversational.

```python
def prioritize_sensor_placement(pipes_df, n_sensors=500, budget=1_500_000, cost_per_sensor=3000):
    pipes_df = pipes_df.copy()

    pipes_df['risk_score'] = (
        (pipes_df['pipe_age_years'] / 50) * 0.4 +
        (pipes_df['historical_leak_count'] / pipes_df['historical_leak_count'].max()) * 0.35 +
        (pipes_df['water_loss_pct'] / 100) * 0.25
    )

    pipes_df['has_sensor'] = pipes_df['sensor_id'].notna()
    candidates = pipes_df[~pipes_df['has_sensor']].copy()

    candidates = candidates.sort_values('risk_score', ascending=False)

    max_sensors = min(n_sensors, int(budget / cost_per_sensor))
    selected = candidates.head(max_sensors)

    return selected[['pipe_segment_id', 'district', 'risk_score', 'pipe_age_years']]
```

## Questions to Ask

1. **"What strategy does this code use for sensor placement?"**
   - Expected: It's a greedy risk-based approach. It scores each pipe segment on a weighted combination of pipe age (40%), historical leak frequency (35%), and current water loss (25%). Then it filters to unmonitored segments, sorts by risk score, and selects the top N segments within budget.

2. **"What's missing from this approach?"**
   - Expected: It doesn't consider spatial coverage — all 500 sensors could end up in one high-risk district, leaving other districts completely blind. No consideration of hydraulic connectivity (one sensor can monitor a range of pipes). No diminishing returns modeling — putting 100 sensors in one district has less marginal value than spreading them out.

3. **"The risk_score combines three features with hardcoded weights. How would you improve this?"**
   - Expected: Learn the weights from data — e.g., train a model to predict leak probability from pipe characteristics, then use predicted probability as the risk score. Or optimize the weights to maximize expected leak detection. The 0.4/0.35/0.25 split is arbitrary.
