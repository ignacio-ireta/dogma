# Pack B — GreenWatt: Renewable Energy Forecasting

## ⚠️ INTERVIEWER ONLY — do not read this file before the exercise.

---

## Overview

| Item | Detail |
|------|--------|
| Domain | Operations / Energy |
| Problem Type | Regression / Time Series Forecasting |
| Client | GreenWatt — renewable energy company, 50 wind farms in Northern Europe |
| Core Issue | Forecast errors costing €8M/year in grid penalties |
| Twist | Acquiring 20 new offshore farms with zero historical data (cold start) |

---

## Opening Script (Read Aloud — ~2 min)

> "Our client is **GreenWatt**, a renewable energy company that operates 50 onshore wind farms across several northern European countries. They sell electricity to the grid, but grid operators require day-ahead power production forecasts — and they penalize forecast errors.
>
> GreenWatt's current forecasting system is a basic statistical model built 5 years ago. It consistently over-predicts output, especially during low-wind periods. They're currently paying about €8 million per year in grid imbalance penalties.
>
> They've asked a consulting team to design a better forecasting system that reduces these penalty costs. They also have a strategic question: they're about to acquire 20 new offshore wind farms, and they need the forecasting system to work for those too."

---

## Information to Reveal ONLY When Asked

### If asked about current forecasting approach:
"They use a simple linear regression based on weather forecasts from the national meteorological service. Wind speed forecast → predicted power output. It was built by an engineer who has since left. No retraining since 2020."

### If asked about data:
"Each farm has sensors recording wind speed, temperature, humidity, and atmospheric pressure every hour. They also have 5 years of historical power output data per farm. Weather forecast data is available from the met service with hourly granularity, 48 hours ahead."

### If asked about the power curve:
**Show Exhibit 2** (wind speed vs power output scatter).

"This is the relationship between wind speed and power output. Notice it's non-linear — there's a cut-in speed below which turbines don't spin, and a cut-out speed above which they shut down for safety. The current model assumes a linear relationship, which partly explains the systematic over-prediction."

### If asked about forecast accuracy:
**Show Exhibit 1** (actual vs predicted time series).

"Here's a one-week sample from one farm. You can see the current forecast consistently overshoots, especially during dips. The MAPE across all farms is about 22%. Industry best practice is around 10-12%."

### If asked about penalty structure:
"Penalties are proportional to the absolute error. Over-prediction and under-prediction are penalized equally. The rate is approximately €30 per MWh of error."

### If asked about the 50 farms:
"They vary quite a bit — different geographies, different turbine models, different terrain. Some coastal, some inland. Performance characteristics differ."

### If asked about the 20 new offshore farms:
"Offshore is different from onshore — higher and more consistent wind speeds, but also different turbulence patterns. GreenWatt has ZERO operational data from these farms. They'll come online in about 8 months."

---

## Challenges to Pose

1. "You proposed [model]. But each farm has different characteristics. Are you training one model or 50? What are the tradeoffs?"

2. "The power curve is clearly non-linear. How does your model capture that? Would a linear model work at all?"

3. "Weather forecasts themselves have errors. Your model is predicting based on predicted wind speed, not actual wind speed. How does that propagate through your system?"

4. "How would you measure if your new model is actually saving them money? What's your evaluation strategy?"

---

## The Twist (Reveal After Initial Structure)

> "Here's the strategic question: GreenWatt is acquiring 20 offshore wind farms that will come online in 8 months. There is zero historical production data from these farms. The turbine models are different from anything in the current fleet. How would you make your forecasting system work for them from day one?"

**Expected insight:** Transfer learning or domain adaptation. Train on the 50 existing farms, then adapt. Use physical models (power curve equations) as a baseline, calibrated with manufacturer specifications for the new turbine models. As real data accumulates, switch to data-driven models. Hierarchical modeling: learn shared patterns across farms, then farm-specific adjustments. Could also use similar offshore farms as proxy data if available in public datasets.

---

## What a Strong Answer Covers

1. **Reframe:** "How do we reduce GreenWatt's forecast errors from 22% MAPE to under 12%, cutting €8M in annual penalties, and design a system that scales to 20 new offshore farms with no historical data?"
2. **Recognizes the non-linearity** — the power curve is NOT linear, current model's fundamental flaw
3. **Model choice for time series:** Gradient boosting with lag features, or LSTM for sequential patterns, or hybrid approach. NOT just "use ARIMA" — this is multivariate with complex physics.
4. **Feature engineering:** Lag features, rolling averages, weather forecast error as a feature, time-of-day/season, turbine-specific characteristics
5. **One model vs many:** Discusses tradeoffs. Likely a hierarchical approach — shared base model with farm-specific fine-tuning.
6. **Cold start solution:** Physics-based baseline + transfer learning + manufacturer specs. Iterative improvement as data accumulates.
7. **Quantifies:** €8M penalties ÷ 22% MAPE = ~€364K per percentage point. Reducing to 12% = 10 points × €364K = €3.6M annual savings. Plus the 20 new farms add another ~€4M penalty risk if not forecasted well.
8. **Evaluation:** Backtesting on historical data (walk-forward validation, not random split for time series). Compare against current model. A/B test on a subset of farms before full rollout.

---

## Timing Guide

| Phase | Target Time | Notes |
|-------|-------------|-------|
| Read scenario | 0:00 - 2:00 | |
| His reframe + questions | 2:00 - 7:00 | Show exhibits when relevant |
| His structure | 7:00 - 20:00 | Challenge at ~12:00 and ~16:00 |
| Drop the twist (offshore) | ~18:00 | This is the big one |
| Ask for synthesis | 22:00 | |
| Synthesis | 22:00 - 25:00 | |
