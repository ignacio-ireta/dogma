---
title: Leakage and walk-forward validation
minutes: 12
min_viable: "Read The system and Where it breaks, then do p01."
---

## The system

A validation score is honest only if the test rows are unseen as production will see them. Leakage is any path
by which the model gets information outside the deployment-time information set: what is observable at the
prediction cutoff. Temporal and target leakage are two violations of that set.

| Leakage type | What leaks | Check that exposes it |
|---|---|---|
| Target | a feature built from the label, such as a row's own target in its encoding | the feature alone predicts the label far too well |
| Temporal | training rows that come after the test rows | the score collapses under walk-forward splits |
| Contamination | the same entity or a near-duplicate row on both sides of a split | grouped folds give a much lower score |
| Preprocessing | scaler, imputer or selector fitted on all rows | refit inside each training fold and compare |
| Label latency | training labels unavailable at the simulated prediction cutoff | audit label availability times and choose the gap |

Exchangeable rows use random folds. Rows in time order use walk-forward splits with cutoff $c_j$: training is
$[0, c_j)$ (expanding) or the last `initial_train` rows (rolling), and the test is $[c_j + g, c_j + g + h)$, with gap $g$ and
horizon $h$.

## Worked example

Take $n = 10$, `initial_train` $= 4$, horizon $h = 2$, step $= 2$, gap $g = 1$, and expanding windows:

| Split | Training rows | Gap row | Test rows |
|---|---|---|---|
| 1 | 0 … 3 | 4 | 5, 6 |
| 2 | 0 … 5 | 6 | 7, 8 |

A third split would test positions 9 and 10, but position 10 does not exist, so the split is not generated.

Target encoding shows the leak in one line. Categories $a, b, a, a$ with targets $1, 5, 3, 2$:

- Leaky (expanding mean including the current row): $1, 5, 2, 2$. The first row encodes its own target $1$ exactly.
- Leakage-safe (earlier rows only): $\mathrm{NaN}, \mathrm{NaN}, 1, 2$.

## Why it works

Walk-forward splits reproduce the condition the model faces: predict the future from the past. Random splits break it,
because a test row sits between its training neighbours and the model interpolates rather than forecasts.

The exercise assumes each earlier target is available before the next prediction. Under that assumption,
using only rows $j < i$ is safe. With delayed labels or a batch forecast horizon, use the actual availability
cutoff instead; earlier row position alone is insufficient. Including $y_i$ puts the label into
its own feature. Ambroise and McLachlan (2002) showed the same mechanism for gene selection on the full data before
cross-validation, and Kaufman et al. (2012) give the general formulation (both cited, not checked against the papers).

Predictive use is not causal use. A variable can improve a forecast and still be wrong to adjust for in a causal
question, such as one measured after treatment. Hernán and Robins (per the Library note) order the work the same
way: identify the estimand before estimating it.

## Signal vs. noise

- **Signal:** where the cutoff sits relative to the features and labels. Anything computed from data after the cutoff
  is a leak, whatever the code looks like.
- **Signal:** the gap. Label latency is a property of the business process, so the gap must match it.
- **Noise:** expanding versus rolling, once the cutoffs are right. Match production.

## Where it breaks

- **Random CV on a series.** A nearest-neighbour model on the time index may interpolate between
  training observations under random folds, although production requires extrapolating beyond its history.
- **Selection before CV.** Selecting features by correlation with all labels makes the held-out labels
  influence the selected features. Refit selection inside each fold, even when all original features are noise.
- **Target encoding.** A singleton category encoded with its full-data target mean reproduces its own
  label exactly. Cross-fitting or an availability-aware historical encoding avoids that direct leak.
- **Long histories and windows.** On real `adult` keys in file order, `native.country` has 5350 United-States rows. A 500-row window instead of the expanding mean changes 1896 of 5959 encodings by more than 0.01. A window, a decay or a batch reset looks harmless on short tests and silently changes the feature on long real series.
- **Duplicates.** Near-identical rows on both sides of a split reward memorisation. Deduplicate or group.
- **Gap too small.** If the label arrives three days late, a zero gap may train on recent labels
  that would not yet have arrived at the simulated prediction time. A row gap assumes a known time interval per row.

**Detection.** A score far above the honest baseline, or far above production, is a symptom. Check feature importance
for identifiers and timestamps. Shuffle the labels and rerun the whole pipeline: above-chance performance can expose target-dependent
leakage. A chance result does not rule out temporal, group or preprocessing leakage. Hyndman and
Athanasopoulos, *Forecasting: Principles and Practice*, is the usual reference for why each training window
must precede the forecast window (cited, not checked against the book).

## Interview angle

**60-second answer.** Features must be computable at prediction time. For forecasting, use walk-forward
splits and an availability-aware gap. Fit preprocessing inside each training fold. A label-shuffle check
can expose target-dependent leakage; audit timestamps and groups separately.

**Follow-ups.**

- *Why does a shuffled random split overstate accuracy on a series?* The model interpolates between neighbours it has
  already seen, so the test is easier than the forecast it is meant to estimate.
- *What is the first place leakage hides?* Preprocessing and selection fitted on all rows before the split.

Full answers are in `interview.md`.

## Connections

- **Requires:** `cml/k-fold-cross-validation` (the folds this topic replaces for ordered data).
- **Unlocks:** `cml/over-and-underfitting-early-stopping`, where a validation set chosen by time must also be protected from
  leakage, and any forecasting or feature-engineering topic.
- **Lineage:** Kaufman et al. (2012) formalise leakage.
