# Pack C — UrbanEats: Food Delivery Demand Prediction

## ⚠️ INTERVIEWER ONLY — do not read this file before the exercise.

---

## Overview

| Item | Detail |
|------|--------|
| Domain | Product / Operations |
| Problem Type | Regression + Optimization |
| Client | UrbanEats — food delivery app, 3M monthly orders in one large city |
| Core Issue | 25% of orders face >45 min delays due to rider supply/demand mismatch |
| Twist | Major music festival in 2 weeks — unprecedented demand spike with no historical data |

---

## Opening Script (Read Aloud — ~2 min)

> "Our client is **UrbanEats**, a fast-growing food delivery platform based in one large city with about 3 million orders per month. They have 800 active riders and partner with 4,000 restaurants across the city.
>
> They have a delivery time problem: 25% of their orders take longer than 45 minutes to deliver, well above their target of 35 minutes average. Customer satisfaction scores have dropped 15 points in the last quarter, and they're losing market share to competitors who deliver faster.
>
> The core issue is a supply-demand mismatch — at peak hours in high-demand districts, there simply aren't enough riders. They're throwing money at rider bonuses during peaks, but it's not sustainable. They've asked a consulting team to design a demand prediction system that helps them position riders proactively, before the rush hits."

---

## Information to Reveal ONLY When Asked

### If asked about current rider allocation:
"Riders choose where to go. There's a basic surge pricing system — higher pay in busy areas — but it's reactive, not predictive. By the time surge kicks in, customers are already waiting."

### If asked about data:
"Two years of order data: timestamps, locations, order values, delivery times, weather conditions. Rider data: home district, vehicle type, working hours, ratings. Restaurant data: location, cuisine type, prep time estimates."

### If asked about demand patterns:
**Show Exhibit 1** (demand heatmap).

"Clear patterns — lunch peak at 12-1 PM, dinner peak at 7-9 PM. Weekends are significantly higher, especially Saturday evenings. Friday nights are also big."

### If asked about the supply-demand gap:
**Show Exhibit 2** (demand vs riders by district).

"Centro and Roma/Condesa have the biggest gaps — demand exceeds rider supply by 30%+. Peripheral districts like Tlalpan are more balanced but have longer delivery distances."

### If asked about weather impact:
"Rain increases demand by about 20% (people don't want to go out) but decreases rider availability by 30% (riders don't want to ride in rain). Heavy rain is a double hit."

### If asked about delivery time breakdown:
"Average delivery time is 38 minutes. Breakdown: 5 min rider acceptance, 8 min rider travel to restaurant, 12 min restaurant prep, 13 min delivery to customer. The biggest variable is rider travel to restaurant — that's where positioning helps."

### If asked about competitor landscape:
"Two main competitors: Rappi and DiDi Food. Rappi has a similar rider model. DiDi has started using demand prediction and pre-positioning — their delivery times dropped 8 minutes in the last quarter."

---

## Challenges to Pose

1. "You want to predict demand 2 hours ahead. How accurate does the forecast need to be for rider pre-positioning to actually work? What's the cost of over-prediction vs under-prediction?"

2. "Riders are independent contractors — you can't TELL them where to go. How does your prediction system translate into actual rider behavior?"

3. "You mentioned weather as a feature. But weather forecasts are unreliable more than a few hours out in this city. How do you handle that uncertainty?"

4. "Centro has the biggest demand-supply gap. Is the solution more riders, or is it something else?"

---

## The Twist (Reveal After Initial Structure)

> "One more thing — in exactly two weeks, a major music festival is happening in the city. Last year's edition wasn't tracked because UrbanEats was smaller then. They expect 80,000 attendees per day for 3 days, concentrated near the Foro Sol area. There's no historical data for an event like this. The CEO wants your system to handle it. What do you do?"

**Expected insight:** Special event handling. Historical data won't help for unprecedented events. Options: (1) Use proxy data from similar events (concerts, sports) to estimate demand multiplier. (2) Pre-position riders near Foro Sol proactively based on event schedule (when sets end = order surge). (3) Establish partnerships with nearby restaurants for increased capacity. (4) Extend surge pricing window. (5) Communicate expected delays to customers proactively. The key insight: for unprecedented events, the model should have an "override" mechanism where operators can inject expert knowledge.

---

## What a Strong Answer Covers

1. **Reframe:** "How do we predict demand 1-2 hours ahead, at the district level, so UrbanEats can pre-position riders and reduce the percentage of >45 min deliveries from 25% to under 10%?"
2. **This is a two-part problem:** Demand prediction (regression) + rider optimization (allocation/optimization). Strong candidates separate these.
3. **Features for demand model:** Hour of day, day of week, weather, district, holidays/events, historical trends, promotional campaigns, restaurant density
4. **Model choice:** Gradient boosting or time series model for demand. Optimization layer on top for rider allocation.
5. **Addresses the incentive problem:** Riders are contractors. Prediction alone doesn't move riders — need incentive design (predictive surge pricing, guaranteed minimum earnings for pre-positioned riders).
6. **Rain paradox:** Demand up + supply down = worst-case scenario. Model needs to account for both sides.
7. **Event handling:** Pragmatic approach combining data proxies, expert overrides, and proactive communication.
8. **Quantifies:** 3M orders × 25% delayed = 750K bad experiences/month. Reducing to 10% = 450K fewer. If each bad delivery has a 5% churn probability and average customer LTV is $500 MXN/month × 12 months = $6,000 MXN, then 450K × 5% × $6,000 = $135M MXN (~$7.5M USD) annual value. Plus reduced rider bonus costs.

---

## Timing Guide

| Phase | Target Time | Notes |
|-------|-------------|-------|
| Read scenario | 0:00 - 2:00 | |
| His reframe + questions | 2:00 - 7:00 | Show exhibits when relevant |
| His structure | 7:00 - 20:00 | Challenge at ~12:00 and ~16:00 |
| Drop the twist (festival) | ~18:00 | |
| Ask for synthesis | 22:00 | |
| Synthesis | 22:00 - 25:00 | |
