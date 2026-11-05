# Case Scenario 5: LogiFreight — Logistics Route Optimization

## Setup (Read Aloud as Interviewer)

Our client is **LogiFreight**, a national freight and logistics company in Latin America with a fleet of 800 trucks operating across 50 distribution centers. They deliver to ~15,000 commercial customers (retail stores, restaurants, warehouses) daily.

Fuel costs have risen 35% in the last two years and now represent 40% of operating expenses — about $120M annually. Their current routing is done by regional dispatchers using experience and basic scheduling software, but there's no optimization.

LogiFreight's CEO believes they're driving 20-30% more kilometers than necessary due to suboptimal routing. They've asked a consulting firm to assess the opportunity and design a solution.

---

## Information to Reveal When Asked

**If asked about data:**
- GPS traces from all 800 trucks (2 years of history)
- Delivery manifests: customer, time window, package count/weight
- Road network data available via OpenStreetMap / Google Maps API
- Customer delivery windows: most require delivery between 6 AM - 2 PM
- Some customers have dock capacity constraints (only 1 truck at a time)

**If asked about current routing:**
"Dispatchers create routes the night before based on orders. They group by geography but don't formally optimize. Each truck typically makes 15-25 stops per day."

**If asked about constraints:**
- Driver hours: max 10 hours driving per day (Mexican law)
- Truck capacity: varies by truck type (8-ton, 12-ton, 20-ton)
- Perishable goods: 30% of deliveries are temperature-sensitive with max 4-hour transit time
- Some roads are restricted at certain hours (city congestion rules)

**If asked about competitor approaches:**
"Major competitors like DHL and FedEx use ML-optimized routing. LogiFreight is behind."

---

## The Twist

"Two months into the project, we discover that 25% of deliveries are 'emergency orders' that come in same-day and disrupt the planned routes. The dispatchers have been padding routes with slack time to accommodate these, which is part of the inefficiency. How do we handle this?"

*Expected insight: Two-layer approach. Strategic (overnight optimization for known orders) + tactical (real-time re-routing for same-day orders). The slack time is a feature, not just waste — it provides flexibility. Dynamic re-optimization during the day as emergency orders come in.*

---

## What a Strong Answer Covers

1. **Reframe:** "How do we reduce LogiFreight's total kilometers driven by 15-25% through optimized routing while respecting delivery windows, driver constraints, and fleet capacity?"
2. **This is an optimization problem, not a prediction problem:** Vehicle Routing Problem (VRP) with time windows — a classic operations research problem.
3. **Approach:** Constraint programming or metaheuristic (genetic algorithm, simulated annealing) for the NP-hard routing problem. Not gradient boosting — this isn't a classification/regression task.
4. **ML component:** Predict delivery time windows more accurately (when will traffic peak?), predict emergency order volume by day/region, predict package volumes for capacity planning.
5. **Pilot strategy:** Start with 2-3 distribution centers. Compare ML-optimized routes vs. dispatcher routes for 1 month. Measure: total km, fuel cost, on-time delivery rate.
6. **Don't ignore the dispatchers:** Their domain knowledge (which roads are bad, which customers are flexible) is valuable. The system augments their expertise.
7. **Quantitative impact:** 20% km reduction on $120M fuel budget = $24M/year savings. Even 10% = $12M/year. Implementation cost of $2-3M = massive ROI.
8. **Emergency orders:** Build in 15% route flexibility. Real-time re-optimization engine that ingests same-day orders and adjusts active routes.
