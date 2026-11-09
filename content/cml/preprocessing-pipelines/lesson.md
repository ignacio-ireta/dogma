---
title: Preprocessing pipelines
minutes: 12
min_viable: "Read The system and Where it breaks, then solve p01."
---

## The system

Every preprocessing step has two phases. **Fit** learns a rule from training rows. **Transform** applies
that rule to any rows. Leakage happens when a transform's rule is learned from rows it will later be
scored on. Each row of the table is a rule learned from training rows.

| Step | Fitted rule | Applied to test row with a missing or new value |
|---|---|---|
| Numeric imputation | training median (or mean) | fill with the training median |
| Standardization | training mean $\mu$ and population std $s$ | $(x-\mu)/s$, with $s=1$ for a constant column |
| Categorical imputation | training mode, ties to the smallest value | a missing value takes the training mode |
| Categorical one-hot | sorted training categories | a value outside them gives an all-zero block |

A **pipeline** runs these steps in order, and a **ColumnTransformer** sends each group of columns to its own
pipeline. Its output concatenates the blocks in the order they were listed. Fit the whole object once on the
training table; use it everywhere else.

## Worked example

Train: age $=(20,\mathrm{NaN},40,30)$, city $=(\text{north},\mathrm{NaN},\text{south},\text{north})$. Test: age
$=(\mathrm{NaN},50)$, city $=(\text{west},\mathrm{NaN})$.

- Age: the observed training median is $30$, so the train gap becomes $30$. The filled column
  $(20,30,40,30)$ has mean $30$ and population std $\sqrt{50}\approx7.07$. The train values become
  $(-1.41, 0, 1.41, 0)$.
- City: the training mode is $\text{north}$ (two rows). Categories are $\{\text{north},\text{south}\}$.
- Test: the missing age becomes $30$, which scores to $0$. The value $50$ scores to $20/\sqrt{50}\approx2.83$.
  The city $\text{west}$ is unseen, so its block is $(0,0)$. The missing city takes the training mode
  $\text{north}$, giving $(1,0)$.

The test row never changed the training mean, the median, the mode or the categories.

## Why it works

Fitting on all rows before splitting makes the test set influence the coordinates the model trains in. The
size and direction of the score change depend on the dataset; the evaluation protocol is compromised
even if the statistic happens not to change. A pipeline makes
the rule explicit: the same fitted object transforms the training rows and then the held-out rows, so a validation loop can fit all steps together. The caller must still pass the correct training rows.

The pipeline is also the unit to validate. Inside cross-validation the whole pipeline is refitted on each
training fold and applied to its held-out fold, so the held-out fold never influences its own preprocessing.
A preprocessing step outside the loop cannot make that guarantee. This is the training-only
resampling rule in [book: Kuhn & Johnson, Applied Predictive Modeling, chapters 3–4]; the
[scikit-learn leakage guide](https://scikit-learn.org/stable/common_pitfalls.html) gives the same protocol.

## Signal vs. noise

- **Signal:** fit on training rows only; keep the fitted object with the model; decide the output column order
  in advance.
- **Signal:** the ties rule for the mode and the constant-column rule for scale are part of the contract.
  Different libraries choose differently, so pin them down.
- **Noise for now:** the speed of sparse encoding, and the choice of which library class carries the name.

## Where it breaks

- **High cardinality.** One-hot encoding a column with thousands of values makes thousands of sparse columns,
  and most of them are rare or unseen at serving time. Hashing or target encoding are alternatives. Target
  encoding is itself a leakage risk: fit it only on training folds.
- **Real rare levels.** On `adult`, `native.country` has 39 training levels in the first 4000 rows, 8 of them seen at most twice. Capping the encoder at the ten most frequent levels would zero the 3.9% of later test rows whose country falls outside that top ten, and a silent cap breaks the contract. Keep every training level.
- **Unseen values.** An all-zero block says "no known category". It does not say the category is rare. Log how
  often unseen values appear before trusting the model on them.
- **All-missing training columns.** A median or a mode does not exist. Raise an error, rather than inventing a
  value.
- **Order drift.** If a column is added or reordered, the fitted weights attach to the wrong features. Store the
  column names with the model.

## Interview angle

Describe the fit/transform split first. Then name the two places where leakage enters with the wrong code:
statistics computed on the full table, and categories built from the test table. The answer closes with the
pipeline: refit it per fold and apply it to the held-out fold. After selection, refit on all development
rows before evaluating once on a separate untouched test set. Follow-ups: what
an unseen category should map to, and why target encoding needs out-of-fold estimates.

## Connections

- **Prerequisites:** `cml/feature-scaling` (the standardization rule), `cml/shuffling-splitting-and-batching-data`
  (the split that defines which rows may be used to fit).
- **Unlocks:** `cml/k-fold-cross-validation` (refitting per fold), `cml/missing-values-and-outliers` (imputation
  and its indicators), and `cml/leakage-and-walk-forward-validation` (what a leak looks like across time).
