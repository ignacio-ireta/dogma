# Interview notes: Confusion matrix and count-based scores

## Q1. The model is 99% accurate. Is it good?

**60-second answer.** Not without the base rate. Accuracy counts every sample equally, so if positives are
1% of the data, a model that predicts "negative" every time also scores 99%. I would first ask for the
prevalence and the all-majority baseline. Then I would report recall and precision for the positive class,
since those say whether the model finds the cases we care about and how many flags are false. If one number
is needed, I would use MCC, which is exactly 0 for a constant predictor by convention, whatever the class sizes.

**Follow-ups.**
- *Which baseline should I compare with?* The all-majority predictor. Its accuracy equals the majority share.
- *Why not just report F1?* F1 ignores true negatives, so it can look fine while specificity is poor. MCC uses all four cells.

## Q2. What is the difference between recall and specificity, and why report both?

**60-second answer.** Recall conditions on the actual positives: $\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$, the share
of real cases found. Specificity conditions on the actual negatives: $\mathrm{TN}/(\mathrm{TN}+\mathrm{FP})$, the share
of real non-cases cleared. The two errors have different costs, so one number cannot describe both. A screening
test with high recall and low specificity sends too many people for follow-up, and the reverse misses cases.

**Follow-ups.**
- *Why not precision instead of specificity?* Precision conditions on predicted positives and changes with the base rate. Specificity stays fixed under a pure prior shift that preserves the class-conditional score distributions and threshold.
- *What is the false positive rate?* $\mathrm{FP}/(\mathrm{FP}+\mathrm{TN}) = 1 -$ specificity.

## Common mistakes

- Transposing the matrix, so rows become predicted labels. FP and FN exchange roles, swapping precision and recall while leaving accuracy, F1 and MCC unchanged.
- Reporting accuracy alone under imbalance, without the base rate or a per-class score.
- Returning NaN for a zero denominator, then letting it flow silently into averages and comparisons.
- Changing which class is "positive" without saying so. Precision and recall change, and so does F1.
