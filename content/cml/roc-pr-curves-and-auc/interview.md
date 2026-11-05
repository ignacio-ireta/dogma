# Interview notes: ROC, PR curves and AUC

## Q1. ROC-AUC is 0.95 on a problem with 1% positives. Is the model useful?

**60-second answer.** Not yet known. ROC-AUC is a rank statistic, the probability that a random positive outranks a
random negative, and it ignores how rare the positives are. With 1% positives, a false positive rate of 1% still
means about as many false alarms as true cases. I would compare average precision with the 1% prevalence baseline,
because AP reacts to prevalence and AUC does not. Then I would check precision and recall at the threshold we would
deploy, since that is the number the business feels.

**Follow-ups.**
- *What would make me prefer PR-AUC?* Rare positives, and a question about the flagged set. State whether the summary is step-wise AP or interpolated PR area.
- *Does a high AUC mean the probabilities are right?* No. AUC is blind to calibration.

## Q2. How do you choose an operating point?

**60-second answer.** Choose it from the constraint or the cost, on held-out data. If the business has a review budget,
cap the number flagged and take the highest recall within that capacity. A precision floor separately limits
wasted reviews. For known error costs, minimise held-out empirical cost, or use the analytic cost threshold on
calibrated deployment probabilities. Report the chosen point alongside the area.
AUC summarises every threshold, and we will deploy only one of them.

**Follow-ups.**
- *Where does the threshold come from?* Held-out data, never the data used for training. Tuning on training data makes the score look better than it is.
- *How do ties affect the choice?* Tied samples move together. A threshold cannot separate them, so the reported precision and recall are for the whole tie.

## Common mistakes

- Reporting ROC-AUC alone on a rare-positive problem, without the prevalence baseline.
- Using trapezoidal interpolation on precision-recall points, which gives a different number from average precision.
- Returning 0.5 when a fold has one class, instead of flagging that AUC is undefined.
- Treating AUC as evidence that predicted probabilities are accurate.
