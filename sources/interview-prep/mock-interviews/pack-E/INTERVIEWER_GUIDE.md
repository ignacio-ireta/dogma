# Pack E — AquaSense: Water Infrastructure Leak Detection

---

## Overview

| Item | Detail |
|------|--------|
| Domain | Operations / Infrastructure |
| Problem Type | Anomaly Detection + Optimization |
| Client | AquaSense — municipal water utility serving 6 million people |
| Core Issue | 30% water loss in the distribution network — $45M/year wasted |
| Twist | Only 10% of the 8,000 km pipe network has sensors — where do you install the next batch? |

---

## Opening Script (Read Aloud)

> "Our client is **AquaSense**, the municipal water utility for a major Latin American city with 6 million residents. They operate 8,000 kilometers of water pipes, ranging from brand new PVC to 50-year-old cast iron.
>
> They have a serious non-revenue water problem: 30% of the water they pump into the system never reaches a customer. Some of that is theft, some is metering errors, but the biggest portion — an estimated 65% of the loss — is physical leaks in the pipe network.
>
> That's about $45 million per year in lost water. Their current approach to leak detection is reactive: they wait for someone to report a visible leak or a pressure complaint, then send a crew. By then, the leak has often been running for weeks or months.
>
> They've installed pressure and flow sensors on about 10% of the network. They want to use this data to detect leaks proactively — and they want to know where to install the next 500 sensors for maximum impact."

---

## Information to Reveal ONLY When Asked

### If asked about water loss patterns:
**Show Exhibit 1** (water loss by district).

"The loss varies dramatically by district. The industrial district is worst at 35%, Centro at 30%. Southern districts are newer and lose less. There's a clear correlation with pipe age."

### If asked about sensor data:
**Show Exhibit 2** (sensor anomaly timeline).

"Here's a real example. Sensor P-047 shows a gradual increase in flow rate and decrease in pressure starting around October 17th. This turned out to be a leak that wasn't reported until November 3rd — 17 days of water loss before anyone noticed. With the right model, we could have caught it within 48 hours."

### If asked about pipe characteristics:
"8,000 km total. Pipe age ranges from 2 to 50+ years. Materials: PVC (30%), cast iron (25%), ductile iron (25%), concrete (20%). Cast iron pipes over 30 years old account for 45% of all reported leaks but only 15% of the network."

### If asked about sensor types:
"Each sensor station measures flow rate (liters/minute), pressure (bar), and temperature (°C). Readings every 15 minutes. Some sensors also detect acoustic signals — the sound of water escaping — but those are expensive and only installed at 50 locations."

### If asked about repair costs:
"Finding a leak costs $2,000 on average (crew, equipment, excavation to confirm). Repairing it costs $5,000-$50,000 depending on pipe size and location. A new sensor station costs $3,000 installed. The math: if a sensor helps find just one leak in its lifetime, it's paid for itself."

---

## Challenges to Pose

1. "Flow rate and pressure fluctuate naturally based on consumer demand. Peak morning and evening usage looks similar to a leak signal. How do you distinguish between the two?"

2. "You mentioned [model]. The pipe network is a physical system — water flows according to hydraulic physics. Should you use a pure ML approach, or incorporate the physics? What are the tradeoffs?"

3. "Cast iron pipes over 30 years old are 45% of leaks but 15% of the network. Should you just replace them all? What's the cost-benefit?"

4. "How do you evaluate your leak detection model when you don't know where ALL the leaks are? You only have data on leaks that were eventually found."

---

## The Twist (Reveal After Initial Structure)

> "The utility board has approved budget for 500 new sensor stations — about $1.5M. Currently sensors cover 10% of the network. Where do you install them to maximize leak detection? What data do you use to decide?"

**Expected insight:** This is an optimization problem on top of the ML problem. Factors for placement: pipe age, material, historical leak frequency, current coverage gaps, district loss rates, hydraulic connectivity (one sensor can monitor a segment of pipe). Possible approach: risk-score every unmonitored pipe segment, prioritize high-risk segments, but also consider spatial coverage (don't cluster all sensors in one area). This connects the predictive model back to actionable infrastructure investment.

---

## What a Strong Answer Covers

1. **Reframe:** "How do we detect leaks within 48 hours using sensor data, and where should we invest in 500 new sensors to maximize coverage of the highest-risk segments?"
2. **This is a two-part problem:** Leak detection (anomaly detection on time series) + sensor placement (optimization)
3. **Anomaly detection approach:** Not just threshold-based. Time-of-day adjusted baselines, rate of change detection, cross-sensor correlation (if neighboring sensor shows pressure drop while yours shows flow increase = likely leak between them).
4. **Handles the "unknown unknowns":** Model can only train on leaks that were found. Active detection will reveal leaks the reactive system missed — need to plan for this.
5. **Physics-informed approach:** Hydraulic models can simulate expected flow/pressure. Deviation from simulation = potential leak. Hybrid ML + physics.
6. **Sensor placement:** Risk-based optimization considering pipe age, material, historical frequency, and spatial coverage.
7. **Quantifies:** 30% loss = ~$45M/year. If proactive detection catches leaks 2 weeks earlier on average, and reduces loss by 10 percentage points → $15M/year savings. Sensor investment of $1.5M pays back in ~5 weeks.

---

## Timing Guide

| Phase | Target Time |
|-------|-------------|
| Read scenario | 0:00 - 2:00 |
| Reframe + questions | 2:00 - 7:00 |
| Structure | 7:00 - 20:00 |
| Drop twist (sensor placement) | ~18:00 |
| Synthesis | 22:00 - 25:00 |
