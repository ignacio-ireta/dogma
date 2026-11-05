# Pack C — Coding Solutions (INTERVIEWER ONLY)

## Problem 1

```python
import pandas as pd

def delivery_performance(orders_path, riders_path):
    orders = pd.read_csv(orders_path)
    riders = pd.read_csv(riders_path)

    order_stats = orders.groupby('district').agg(
        total_orders=('order_id', 'count'),
        avg_delivery_time=('delivery_time_min', lambda x: round(x.dropna().mean(), 1)),
        pct_delayed=('delivery_time_min', lambda x: round((x.dropna() > 45).mean() * 100, 1)),
        avg_order_value=('order_value_mxn', lambda x: round(x.dropna().mean(), 2)),
    ).reset_index()

    rider_counts = riders.groupby('home_district').size().reset_index(name='rider_count')
    rider_counts.rename(columns={'home_district': 'district'}, inplace=True)

    result = order_stats.merge(rider_counts, on='district', how='left')
    result['rider_count'] = result['rider_count'].fillna(0).astype(int)
    result['orders_per_rider'] = round(result['total_orders'] / result['rider_count'].replace(0, float('nan')), 1)

    return result.sort_values('pct_delayed', ascending=False)
```

## Problem 2

```python
def weather_peak_analysis(orders_path):
    orders = pd.read_csv(orders_path)

    peak_hours = [12, 13, 19, 20, 21]
    orders['is_peak'] = orders['hour'].isin(peak_hours)

    result = orders.groupby(['weather', 'is_peak']).agg(
        order_count=('order_id', 'count'),
        avg_delivery_time=('delivery_time_min', lambda x: round(x.dropna().mean(), 1)),
        pct_delayed=('delivery_time_min', lambda x: round((x.dropna() > 45).mean() * 100, 1)),
        avg_order_value=('order_value_mxn', lambda x: round(x.dropna().mean(), 2)),
    ).reset_index()

    return result.sort_values('avg_delivery_time', ascending=False)

# Bonus: Worst = Heavy Rain + Peak. Best = Clear + Off-Peak.
# Difference quantifies the compounding effect of weather + demand pressure.
```
