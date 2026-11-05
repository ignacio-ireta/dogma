# Predictive maintenance system, debrief

## Reframe

Which compressors should crews inspect each week, so that the most unplanned failures are caught with at
least ten days of warning, within the 15 inspections a week the crews can do? Success is net value per week:
39,000 dollars for each failure caught with lead time, minus 1,200 dollars for each inspection. ROC AUC
measures the model. It does not measure the decision.

## Ask

Ask how a failure is recorded, because the label depends on it. Ask for the parts lead time, which sets the
earliest useful alert, the crew capacity, what an inspection and a repair cost, and how overhauls and moves are
recorded. Ask what happens to a flagged compressor, since early repairs hide failures from the
record. Each answer sets the label, the budget or the evaluation. The most important is the last one: if
early repairs erase failures, the labels describe past alerts as much as past faults.

## Structure

Start with the label: failure starts within 24 days. Rows in the last 24 days before the data cutoff are
unknown, so they are dropped, not marked negative. Cut each unit's history at overhauls and moves, and treat
those points as censoring. A discrete-time hazard model is the principled option; a classifier with censored
rows removed is the fast one.

Features: rolling statistics per sensor over 1, 7 and 28 days, each unit's deviation from its own 90-day
baseline, days since the last overhaul, and site. Use a logistic baseline first, then gradient-boosted trees.
Handle the 0.33 percent positive rate with class weights and, above all, with the alert budget. Resampling
changes the training metrics without changing the list the crew receives.

Split by time: train on the first thirty months and test on the last six. Evaluate the ranked list at each
weekly budget: failures caught with ten days of lead, inspections per week, and net value. Check calibration
before turning chances into dollars. Break-even precision is about 3 percent, so a low-precision list can pay
if it is cheap to inspect, and an AUC gain of a few points may not change it at all.

## Synthesize

At 10 alerts a week, the held-out half-year caught 17 of 30 failures with lead time, from 260 inspections.
Each catch is worth 39,000 dollars, the 48,000-dollar unplanned failure less the 9,000-dollar planned repair
that replaces it. Over a year that is about 34 failures caught, worth about 1.3 million dollars, less about
0.6 million of inspections: roughly 0.7 million dollars a year of net value before model and sensor costs.
Between 10 and 15 a week, the average extra alert loses money: 3 more failures caught for 130 more
inspections is a marginal precision of 2.3 percent, below the 3.1 percent break-even, even though the crews
could do 15. Ten is the best budget on this grid; the pilot should measure 11 to 14 before the budget is fixed.
The decision is the budget, not the model.

Pilot the list in shadow mode at two sites for eight weeks, recording every inspection outcome, and compare
caught faults with the current walks. The twist shows that the list can grow without a cap: going from 10 to 30 a
week implies the alerts were set by a score threshold, not by the budget. Enforce the cap in code, rank by score
and cut at the budget, and watch alert volume against inspection outcomes every week. Before any scale-up,
reconcile the sensor register with the alert log; the sensor swap is the first thing that would break the pilot. Only after a clean record should anyone
consider changing the overhaul interval.
