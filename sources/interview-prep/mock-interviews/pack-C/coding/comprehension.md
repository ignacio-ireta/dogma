# Code Comprehension — What Does This Code Do?

**Format:** Show this code to the candidate. Ask the questions below. Conversational — no coding required.

```python
def optimize_rider_allocation(demand_forecast, rider_pool, max_distance_km=5):
    allocation = {}
    available = rider_pool.copy()

    demand_sorted = demand_forecast.sort_values('predicted_orders', ascending=False)

    for _, district in demand_sorted.iterrows():
        district_name = district['district']
        needed = int(np.ceil(district['predicted_orders'] / 8))

        nearby = available[available['home_district'] == district_name]

        if len(nearby) >= needed:
            assigned = nearby.head(needed)
        else:
            assigned = nearby.copy()
            remaining_need = needed - len(nearby)
            other = available[available['home_district'] != district_name]
            assigned = pd.concat([assigned, other.head(remaining_need)])

        allocation[district_name] = assigned['rider_id'].tolist()
        available = available[~available['rider_id'].isin(assigned['rider_id'])]

    return allocation
```

## Questions to Ask

1. **"Walk me through the logic of this function."**
   - Expected: It's a greedy rider allocation algorithm. It sorts districts by predicted demand (highest first), calculates how many riders each district needs (orders / 8 deliveries per rider), then assigns riders — preferring local riders first, then pulling from other districts if needed. Once assigned, riders are removed from the available pool.

2. **"What's the problem with this greedy approach?"**
   - Expected: It prioritizes high-demand districts, which means they get first pick of local riders. Lower-demand districts might get stuck with riders from far away, even if a more balanced allocation would reduce total travel time. It also doesn't consider rider travel distance — just whether the rider is from the same district or not. The `max_distance_km` parameter is defined but never used.

3. **"How would you improve this?"**
   - Expected: Use an optimization approach (e.g., linear programming) that minimizes total rider travel time while meeting demand constraints. Consider rider capacity based on vehicle type (bicycles vs motorcycles). Account for rider preferences and historical performance by district. The unused `max_distance_km` parameter should actually filter which "other district" riders are eligible.
