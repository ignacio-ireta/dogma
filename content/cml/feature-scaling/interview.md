# Interview notes — Feature scaling

## Q1. When does feature scaling matter?
**60-second answer.** Start with the model mechanism. Distances compare coordinates,
optimization follows gradients, and regularization penalizes coefficients in their
chosen units. Scaling can materially change those operations. Exact threshold
trees preserve row partitions under strictly increasing transforms, so they usually
need less preprocessing for units. Fit all statistics on training data and deploy
them with the model.
**Follow-ups.** Does scaling fix outliers? No; mean and standard deviation are
outlier-sensitive. Does it create normal data? No; distribution shape is unchanged.
What about categorical ids? Their numeric labels are not meaningful distances.

## Q2. Why is fitting a scaler before cross-validation leakage?
**60-second answer.** The validation fold then influences the coordinate system
used during training. Labels need not be involved for information to leak.
Each fold needs a freshly fitted scaler using only that fold's training rows.
After selecting the model, refit the pipeline on all development data and evaluate
once on the untouched test set.
**Follow-ups.** Does leakage always improve the score? No; its direction and size
depend on the dataset, but it violates the intended independence. How to prevent
it? Put preprocessing and estimation in one pipeline.

## Common mistakes
- Scaling each row when the contract asks for per-feature statistics.
- Using the held-out batch's mean or a sample standard deviation.
- Treating every small nonzero scale as a constant.
- Zeroing out unseen values of a feature that was constant during training.
