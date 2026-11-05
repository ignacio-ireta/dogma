# Interview notes: K-fold cross-validation

## Q1. How would you estimate how well this model generalises?

**60-second answer.** I would use k-fold cross-validation. Each row is held out exactly once, and the model is
trained on the other rows for each fold. The estimate is the mean of the k fold scores, and I report their
spread as well. Averaging over k folds is less sensitive to one lucky split than a single hold-out. The folds
must follow the structure of the data: stratified when classes are rare, grouped when rows share an entity, and
time-ordered when the rows form a series. A random split of time-ordered data lets the model see the future.

**Follow-ups.**
- *Why not leave-one-out?* It fits n models on almost identical training sets, so its errors are strongly
  correlated, and it costs n fits. Five or ten folds are usually the better trade-off.
- *How do you pick k?* Smaller k gives each model less training data, which can make the estimate pessimistic. Larger k
  costs more fits. It is a compute and bias trade-off, not a statistical rule.

## Q2. You tuned hyperparameters with 5-fold CV and report that CV score. Is it honest?

**60-second answer.** Not quite. The score is optimistic, because I selected the best of many candidates by
their score on the same folds. The selection picks up the noise in those folds. The honest version is nested
CV: an inner loop chooses hyperparameters inside each outer training set, and the outer loop scores the whole
procedure on rows the tuning never saw. Alternatively, I keep a test set that no tuning step touches, and score
it once.

**Follow-ups.**
- *How much does it matter?* It depends on sample size, search size and dependence among candidates. Choosing the largest noisy
  score tends to select positive noise even when the candidates have identical true performance.
- *Does nested CV cost much?* It multiplies the fits by the number of outer folds times the inner search size.

## Q3. What changes when the rows are imbalanced or grouped?

**60-second answer.** With two positives in twenty rows and five folds, at least three test folds contain no
positive, so per-fold recall is undefined there. I would use stratified folds, and report the pooled out-of-fold
metric. If several rows come from one patient, user or device, random folds put the same entity on both sides of
the split. That inflates the score. I would use grouped folds, so an entity lands in one side only.

**Follow-ups.**
- *Why report the out-of-fold metric?* It uses every row exactly once for scoring, which can make rare-class metrics
  defined when the full dataset contains the required classes. Fold-score spread is not a confidence interval.
- *Is stratification enough for grouped data?* No. Stratification balances labels, not entities.

## Common mistakes

- Reporting the best CV score after tuning on the same folds, with no outer loop or untouched test set.
- Fitting the scaler, imputer or feature selector on all rows before splitting.
- Shuffling time-ordered data with a random split.
- Random folds on repeated measures, so one entity appears in both training and test.
- Quoting the mean of fold scores without the spread across folds.
