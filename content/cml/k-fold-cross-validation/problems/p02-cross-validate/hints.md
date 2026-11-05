## Hint 1 — concept

Evaluation is one loop over the folds you are given. In each fold, the model learns only from the
training rows and is scored only on the held-out rows. The function summarises the fold scores; it does
not choose the folds, and it must never let a test row reach `fit_predict`.

## Hint 2 — formula / approach

For fold `j`, the score is `metric(y[test_j], fit_predict(X[train_j], y[train_j], X[test_j]))`.
The summary is the average of the `k` scores, and the sample standard deviation with `k - 1` degrees of
freedom: `sqrt(sum((s - mean)**2) / (k - 1))`.

## Hint 3 — code shape

Validate the lengths, the number of folds and each index array first. Then loop over the folds in order:
index `X` and `y` with the train indices, call `fit_predict`, check that it returned one value per test
row, and append `float(metric(y[test], predictions))`. Finish with `np.std(scores, ddof=1)`.
