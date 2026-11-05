# Interview notes: Handling class imbalance

## Q1. Your data is 1% positive. What do you do?

**60-second answer.** I start with the metric, because accuracy would reward a model that never flags anything. I
would report average precision next to the 1% baseline, or recall at a precision floor the business can afford. Then
I set the threshold from the costs of each error, using calibrated probabilities, since the optimal rule is a
simple function of the two costs. I change the training only if the model is short of minority signal. Class
weights are the least invasive option. Whatever I change, I check calibration at the real base rate, because
weights and resampling both move the probabilities.

**Follow-ups.**
- *Do you resample first?* Not as a first move. It changes the probabilities and it risks leakage if done before the split.
- *How do you correct probabilities after weighting?* Under the proper-loss, class-only weighting idealisation, use deployment prior odds divided by the effective weighted training prior odds, then check deployment calibration.

## Q2. Why does resampling break calibration, and how do you fix it?

**60-second answer.** Resampling changes the proportion of classes the model sees, so the probabilities it outputs
refer to the resampled proportions, not to the real base rate. Random oversampling also repeats points, so a flexible
model can overfit them. The fix for the probabilities is the prior-shift correction: multiply the odds by the ratio
of deployment prior odds to sampled training prior odds, then convert back. This assumes unchanged feature
distributions within each class; SMOTE can violate that assumption. It does not fix overfitting, so the model still needs validation on
data that keeps the true base rate.

**Follow-ups.**
- *Is SMOTE safe?* It adds interpolated points. It assumes the minority class is locally connected, and it should be applied only inside training folds.
- *Does a threshold need the correction?* A cost threshold on calibrated probabilities does. Without it, the threshold refers to the wrong base rate.

## Common mistakes

- Choosing the model by accuracy, which favours predicting the majority class.
- Resampling the whole dataset before splitting, so synthetic or duplicated points leak into validation.
- Reporting probabilities from a weighted or resampled model as if they were calibrated to the real base rate.
- Using a threshold of 0.5 when the two errors have very different costs.
