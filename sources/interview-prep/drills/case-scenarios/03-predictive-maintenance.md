# Case Scenario 3: SteelNord — Predictive Maintenance

## Setup (Read Aloud as Interviewer)

Our client is **SteelNord**, a major European steel manufacturer with 4 plants across several countries. They operate 120 industrial furnaces — the core of their production.

Currently, furnaces are maintained on a fixed schedule: full inspection every 90 days, regardless of condition. This costs approximately €2M per year across all plants. But they're still experiencing unplanned breakdowns — about 15 per year — which cost €150K each in lost production, emergency repairs, and delayed orders. That's €2.25M in breakdown costs on top of the €2M in scheduled maintenance.

SteelNord wants to shift from time-based to condition-based maintenance. Can we predict which furnaces are likely to fail and when?

---

## Information to Reveal When Asked

**If asked about sensor data:**
"Each furnace has 40 IoT sensors recording every 5 minutes: temperature (multiple zones), pressure, vibration, gas flow rates, power consumption, exhaust composition."

**If asked about historical data:**
"3 years of sensor data + maintenance logs. Every maintenance event is logged: type (scheduled/emergency), duration, parts replaced, root cause category. About 180 failure events total."

**If asked about failure patterns:**
"Most failures are in heating elements (40%), refractory lining (30%), and control systems (30%). Heating element failures usually show a gradual temperature drift over 2-3 weeks before failure."

**If asked about production impact:**
"Each furnace runs 24/7. Downtime costs about €10K per hour in lost production. An unplanned failure takes 3-5 days to repair. A planned maintenance stop takes 1-2 days."

**If asked about the 90-day cycle:**
"Some furnaces could probably go 120 days without issues. Others have failed at 45 days. The fixed schedule doesn't account for actual condition."

---

## The Twist

"The operations manager is skeptical of ML. He's been maintaining furnaces for 25 years and says he 'can tell by the sound' when something's wrong. How would you get his buy-in?"

*Expected insight: Don't fight domain expertise — incorporate it. His heuristics become features. Validate model predictions against his intuition. Frame ML as augmenting his expertise, not replacing it. Pilot on a few furnaces first and show results.*

---

## What a Strong Answer Covers

1. **Reframe:** "How do we predict furnace failures 2+ weeks in advance so SteelNord can schedule maintenance proactively — reducing both breakdown costs and unnecessary scheduled maintenance?"
2. **This is a time series / survival analysis problem:** Not just classification. When will it fail? Remaining useful life (RUL) estimation.
3. **Feature engineering from sensor data:** Rolling averages, rate of change, deviation from baseline, cross-sensor correlations (temperature-vibration relationship)
4. **Model options:** Survival analysis (Cox proportional hazards), gradient boosting with engineered features, or LSTM if sequential patterns are critical. Start with gradient boosting on engineered features (simpler, interpretable, proven).
5. **Domain expert integration:** The operations manager's "sound" heuristic = vibration patterns. Encode his expertise as features. Make him an ally, not an adversary.
6. **Quantitative impact:**
   - Reduce unplanned failures from 15 to ~3/year: 12 × €150K = €1.8M saved
   - Extend some maintenance cycles: ~20% reduction in scheduled maintenance = €400K saved
   - Total value: ~€2.2M/year savings vs implementation cost of maybe €300-500K
7. **Implementation:** Start with 10 furnaces (pilot), validate against maintenance logs, expand if successful. Real-time dashboard for maintenance team.
