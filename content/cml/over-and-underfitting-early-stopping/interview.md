# Interview notes: Over- and underfitting, early stopping

## Q1. The model is excellent on training data and poor on validation. What do you do?

**60-second answer.** That suggests overfitting, although I also check distribution shift and metric consistency. I
check the sizes first. If the training set is small relative to the features, I add rows or reduce features. Then I
pick a fix that matches the cause: lower capacity, such as a smaller degree or a shallower tree; regularisation, such
as an L2 penalty that shrinks weights toward zero; or early stopping, which limits how far the optimiser moves. I
choose among them with cross-validation, and I confirm the gap closes on a held-out test set.

**Follow-ups.**
- *How do you tell underfitting from overfitting?* Underfitting shows high training and validation error with a small
  gap. Overfitting shows low training error and high validation error.
- *Does more data always help?* It often reduces estimation variance, but does not generally fix a misspecified model; finite-sample bias can also change.

## Q2. How would you decide when to stop training?

**60-second answer.** I watch the validation loss, and stop when it has not improved for a set number of epochs. That
is patience. A min_delta defines what counts as an improvement, so that noise does not reset the counter. I keep the
best checkpoint, not the last one. Then I report a test score that the stopping decision never saw. Early stopping
can help with late noise fitting or instability, and it can cost accuracy on runs that improve slowly.

**Follow-ups.**
- *Why report the win rate too?* A mean difference can be tiny while the checkpoint wins on fewer than half the
  seeds. The win rate across seeds shows whether the typical run benefits.
- *What about the validation set?* Early stopping uses it to choose the epoch, so it is no longer an unbiased
  estimate. Keep a test set.

## Q3. When does early stopping hurt?

**60-second answer.** When the validation curve is noisy, a small patience stops too early. When the curve is flat for
a long time, as in warm-up or on a plateau, it can stop before the model has learned anything. When the validation set
is small or unrepresentative, the stop epoch is itself noise. And when training is already well regularised, the
validation loss may never rise, so stopping gains nothing. I set patience from the typical length of a plateau.

**Follow-ups.**
- *Is the validation curve always U-shaped?* No. Over-parameterised models can show a second descent, so a rising
  curve does not always mean stop. See the bias-variance lesson.
- *What do you report?* The test score at the chosen checkpoint, plus the spread across seeds.

## Common mistakes

- Selecting the degree, the stop epoch or the regularisation strength on the validation set, then reporting that
  validation score as the result.
- Taking the last checkpoint and ignoring the best one.
- Setting patience to one, which stops at the first noisy uptick.
- Judging early stopping by the mean test error over seeds alone.
- Using a validation set that is not representative of the deployment period.
