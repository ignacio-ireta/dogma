---
title: K-fold cross-validation
minutes: 12
min_viable: "Read The system and Where it breaks, then do p01."
---

## The system

Ordinary $k$-fold cross-validation turns one hold-out split into $k$ splits, so every row is held out exactly once.
The following grouped and time-ordered variants impose additional constraints; time windows need not cover every row.

| Fold structure | Use it when | Rule |
|---|---|---|
| Unshuffled $k$-fold | rows are exchangeable and already in no meaningful order | consecutive blocks of positions |
| Shuffled $k$-fold | rows are exchangeable but stored in some order (for example by label) | blocks of a seeded permutation |
| Stratified $k$-fold | classes are imbalanced | each fold keeps the class proportions |
| Grouped $k$-fold | several rows share an entity (patient, user, device) | a group sits on one side of every split |
| Time-ordered | rows form a series | train on the past, test on the future |

With $q, r = \operatorname{divmod}(n, k)$, the first $r$ test folds hold $q + 1$ rows and the rest hold $q$. Fold $j$
tests on $T_j$ and trains on its complement. Here the score is the unweighted mean of scalar fold scores $s_1, \dots, s_k$, and its spread
is their sample standard deviation (divisor $k - 1$). Fold-score spread is descriptive: overlapping training sets
make scores dependent, so dividing it by $\sqrt{k}$ does not generally yield a valid standard error.

## Worked example

Take $n = 10$ and $k = 3$. The fold sizes are $4, 3, 3$, because $10 = 3 \cdot 3 + 1$:

| Fold | Test rows | Training rows | Training size |
|---|---|---|---|
| 1 | 0, 1, 2, 3 | 4 … 9 | 6 |
| 2 | 4, 5, 6 | 0 … 3, 7 … 9 | 7 |
| 3 | 7, 8, 9 | 0 … 6 | 7 |

Every row is tested once. If the fold scores are $0.80, 0.70, 0.90$, the reported score is $0.80$ and the sample
standard deviation is $0.10$.

## Why it works

A single hold-out set can be sensitive to which rows it contains. Ordinary $k$-fold CV evaluates every
row once and averages scores from several fitted models. This can reduce sensitivity to one split, but
the folds are correlated and there is no universal variance-reduction guarantee. The mean of fold means
also differs from a pooled per-row loss when test folds have unequal sizes. See
[ESL, chapter 7](https://hastie.su.domains/ElemStatLearn/).

The price is training size. Each fold model sees $(k - 1)/k$ of the data, so the estimate can be pessimistic for a learner that improves with more training data.
Larger $k$ reduces that bias but costs more fits. Kohavi (1995) recommended stratified 10-fold CV as a default.

## Signal vs. noise

- **Signal:** the fold structure. Stratify, group or order the folds to match the data. This choice decides whether the
  estimate is honest, and the arithmetic of the split does not.
- **Signal:** the spread across folds. A mean without its spread hides how unstable the estimate is.
- **Noise:** whether $k$ is 5 or 10, which is rarely decisive. Where the remainder sits is bookkeeping.

## Where it breaks

- **Imbalance.** With two positives in twenty rows and five folds, at least three test folds contain no positive, so
  per-fold recall is undefined there. The chance that a given fold of four rows has no positive is
  $\binom{18}{4} / \binom{20}{4} = 3060 / 4845 \approx 0.632$. Stratify, and report the pooled out-of-fold metric.
- **Repeated entities.** Random folds over repeated measures put one patient in both training and test. The score
  rises for reasons unrelated to generalisation. Use grouped folds.
- **Time.** Shuffling a series lets the model train on the future. See the leakage lesson for walk-forward splits.
- **Selection.** Tuning on the same folds and reporting the best score is optimistic. Even when candidates have the same true performance, choosing the largest noisy
  score favors positive estimation errors. Nested CV separates selection from evaluation: the inner loop tunes, the outer loop scores the whole procedure (Varma and Simon, 2006).

## Interview angle

**60-second answer.** I would use k-fold cross-validation, because every row is held out once and the estimate averages
over $k$ partitions rather than one lucky split. I report the spread too. The folds must match the data: stratified for
rare classes, grouped for shared entities, time-ordered for series. If I tune hyperparameters, the honest score needs
nested CV, or a test set the tuning never touched.

**Follow-ups.**

- *Why not leave-one-out?* It fits $n$ models on nearly identical training sets. Those errors are strongly correlated,
  so the averaged estimate can be noisy, and the cost is $n$ fits.
- *Does nested CV cost much?* It multiplies the fits by the outer folds times the inner search size. That is worth it
  whenever the reported number matters.

Full answers are in `interview.md`.

## Connections

- **Requires:** `cml/expectation-and-variance` (the spread of an average) and `cml/regression-metrics-and-robust-losses`
  (the metrics that each fold reports).
- **Unlocks:** `cml/leakage-and-walk-forward-validation` (time-ordered folds and leakage), `cml/over-and-underfitting-early-stopping`
  (choosing complexity on validation folds) and `cml/hyperparameter-search`.
- **Lineage:** Arlot and Celisse (2010) survey the procedures. ESL chapter 7 covers the bias and variance of CV.
