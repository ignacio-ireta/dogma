# Interview notes — Predictive maintenance system

## Q1. How would you define the label for "about to fail"?
**60-second answer.** I start from what the records can count: an unplanned stop coded as a component failure.
The label is failure within a fixed horizon, chosen so that the useful alert window, from the parts lead time out
to the horizon, is wide enough to act on. I drop the last part of the data where the outcome is not yet known, and I treat
overhauled or moved units as censored, not negative. The label is fixed before the first model.
**Follow-ups.** Why not label only the failure day? The list has to cover the days before a failure, not
just the day itself. What if the horizon is shorter than the lead time? The alerts arrive after the parts
could have been fitted, so the list cannot change the outcome.

## Q2. The model has an AUC of 0.93, but the maintenance team sees no savings. Why?
**60-second answer.** AUC ranks all compressors; it does not say how many to inspect. Savings depend on how many
of the top of the list are real faults caught with lead time, against the cost of each inspection. I would
report failures caught with lead time at each weekly budget and net value per week, then set the budget where
the marginal alert stops paying.
**Follow-ups.** What is break-even precision? The inspection cost divided by the value of a caught failure, about
3 percent here. How would you know the list works? Shadow mode against the current walks, measured on caught faults.

## Q3. A sensor is replaced mid-year. What do you do?
**60-second answer.** I stop trusting raw levels from the affected sites until I know what changed. I check the
asset register against the alert log, compare the two sensor generations on an overlap period, and move features
toward each unit's own baseline so that a change of scale cancels out. Units whose inputs fail a drift check stay
off the list, and I add monitoring on alert volume and on inspection outcomes.
**Follow-ups.** Can you just retrain? Retraining on mixed data learns the new scale as if it were a signal. Why
watch alert volume? A jump in alerts with no faults found is the earliest symptom, usually before any metric moves.

## Common mistakes
- Labelling the last weeks of data as negative because their outcome is not yet known.
- Splitting daily windows at random, which leaks the same unit across train and test.
- Choosing a threshold by accuracy or F1, not by the crew budget and the cost of each error.
- Quoting AUC as the business result.
