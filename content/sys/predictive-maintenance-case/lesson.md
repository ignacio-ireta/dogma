---
title: Predictive maintenance system
minutes: 12
min_viable: "Read The system and Where it breaks, then the brief and the twist."
---

## The system

Predictive maintenance is a ranking problem under a crew budget. Each day a model scores every asset
with the chance it fails inside a fixed window. Crews inspect the top of the list, and the list is cut
to an alert budget: the number of inspections worth paying for, which can never exceed the crew's
capacity. Four definitions come before any model:

1. **Failure.** Name the event the records can count: an unplanned stop, coded as a component failure.
   Planned work and faults fixed within a shift are not failures.
2. **Label.** For each asset-day, $y = 1$ if a failure starts within the next $H = 24$ days, else $y = 0$.
3. **Lead time.** Parts take $L = 10$ days to arrive. An alert is actionable only if it fires at least
   $L$ days before the failure, so the useful window is 10 to 24 days ahead.
4. **Decision value.** A caught failure saves money, and each inspection costs money. The budget is set
   where the marginal alert stops paying.

## Worked example

A fleet of 1,200 compressors has 60 unplanned failures a year. The failure rate is
$60 / (1200 \times 365) \approx 0.00014$ per asset-day. Each failure makes the 24 days before it
positive, so about 0.33 percent of asset-days are positive. A model that never flags anything is
99.7 percent accurate.

Costs: an unplanned failure costs 48,000 dollars. A planned repair after a confirmed fault costs 9,000.
An inspection costs 1,200. A missed failure therefore costs 40 times a false alarm, which is why a
low-precision list can still pay. A caught failure is worth $48{,}000 - 9{,}000 = 39{,}000$ dollars. An alert
pays when its chance of catching one exceeds $1{,}200 / 39{,}000 \approx 3.1$ percent. Over a
held-out half-year, the list at 10 alerts a week catches 17 of 30 failures with lead time, from 260
inspections. Going from 10 to 15 alerts a week adds 3 caught failures for 130 more inspections, a
marginal precision of 2.3 percent, below break-even.

## Why it works

Rank with a model, then set the budget from the cost of each error. AUC describes the ranking;
the budget describes the decision. Use a logistic baseline first, then gradient-boosted trees on
rolling sensor statistics and each unit's deviation from its own baseline.

A longer horizon makes more failures catchable but makes each alert less precise, and the lead-time floor
makes the short end useless. Units that are overhauled, moved or retired are censored: their outcome after
that date is unknown, not zero. A discrete-time hazard model handles censoring directly; a plain classifier
needs those rows dropped or marked unknown (Kleinbaum and Klein, on censoring). Split by time, not at
random, because daily windows from one unit overlap and leak across a random split.

## Signal vs. noise

**Signal:** a sustained drift in vibration or bearing temperature relative to the unit's own baseline,
rising motor current at fixed discharge pressure, and the time since the last overhaul.

**Noise:** absolute readings that differ mainly by installation, and small gains in AUC that do not change
the list the crew inspects.

## Where it breaks

- **Censoring.** Dropping overhauled or moved units keeps only the units that stayed, which biases the rate.
- **Intervention feedback.** Early repairs after an alert remove failures from the record. Labels then
  describe past alerts as much as past faults. Check the inspection log, not only the failure log.
- **Sensor change.** A new sensor scale moves every raw reading. The first symptom is an alert count that
  jumps while inspections find nothing.
- **Metric drift.** AUC can rise while the list stops paying. Track caught failures with lead time per
  week and net value.

## Interview angle

Sixty-second shape: "A failure is a coded unplanned stop. The label is failure within 24 days, and only
alerts at least 10 days early are useful. I would rank with a time-split model against a logistic
baseline, cut the list to the crew's weekly budget, and measure net value per week on a holdout. Expect
follow-ups on censoring and on why AUC is not the business result."

## Connections

Builds on precision, recall and thresholds. It shares the decision framing of the fraud and churn cases,
where the cost of each error sets the threshold. Its new piece is censoring, which links to survival
analysis.
