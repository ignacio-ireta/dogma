# Interview notes: Leakage and walk-forward validation

## Q1. Your model's score jumped after you added one feature. What do you check?

**60-second answer.** I assume leakage until I have ruled it out. First I check whether the feature could be
computed at prediction time: a feature built from the target, from a later timestamp, or from an aggregate that
includes the test rows fails that test. Then I look at feature importance, since an identifier or a timestamp
at the top is a warning. Finally I run a label-shuffle check: permute the labels, run the whole pipeline, and
see whether it still scores above chance. Persistent above-chance results can expose target-dependent leakage, but chance results do not rule out other leaks. Only after that do I believe the gain.

**Follow-ups.**
- *What is the shuffle check measuring?* Whether the pipeline can produce signal from labels that carry none. A
  correct pipeline has no expected signal, though finite scores fluctuate; temporal and group leakage still need separate audits.
- *Which leakage is most common in practice?* Preprocessing or feature selection fitted on all rows before the
  split, and target encodings computed over the full data.
- *Is a strongly predictive feature safe to use?* Not on that fact alone. Prediction and causal adjustment answer
  different questions: a variable affected by the treatment or the outcome can predict well and still bias a
  causal estimate if it is adjusted for.

## Q2. How do you validate a model that predicts next week's demand?

**60-second answer.** I use walk-forward validation. Each split trains on the past and tests on a window of the
same length as the forecast horizon, and the window moves forward by a step. The gap must cover the label latency:
if a week's demand arrives three days after the week ends, the last training label must already be available when the
first test prediction is made, so I set the gap from that arrival lag and not from the horizon. I
choose expanding or rolling windows to match how much history the production model really uses. Random folds
would let the model learn from the future, and they would overstate accuracy.

**Follow-ups.**
- *Expanding or rolling?* Expanding uses all history. Rolling adapts faster when the process drifts, at the cost
  of fewer training rows.
- *Why not use the last split only?* One window is one draw of the error. Several windows show how stable the
  error is over time.

## Q3. How do you encode a high-cardinality categorical feature without leaking?

**60-second answer.** When earlier labels have arrived, I replace each category by the mean target of available earlier rows with that category, and use NaN
for the first occurrence. For a fixed multi-step forecast, I freeze the mapping at the forecast cutoff. The encoding of a row then never sees its own target or anything after it. In a
random-split setting I compute the encoding out-of-fold, inside each training fold only. I also smooth rare
categories toward the global mean, so that a category seen once does not produce an extreme value.

**Follow-ups.**
- *Why does the first occurrence need NaN?* It has no earlier rows of its category, and this exercise chooses that convention. A training-only global prior is another valid policy.
- *What about the smoothing constant?* It trades bias for variance on rare categories. I choose it by
  validation, inside the training data only.

## Common mistakes

- Fitting a scaler, imputer, encoder or feature selector on all rows before splitting.
- Encoding a category with the mean of every row, including the row being predicted.
- Shuffling time-ordered rows before splitting them.
- Ignoring label latency, so the features for a test row are built from data that arrives after its label.
- Reporting a single split's score as if it were the expected performance over time.
