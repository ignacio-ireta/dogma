---
title: Random forests and feature importance
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A **random forest** is an average of $B$ trees, each grown on its own bootstrap sample.

- **Bootstrap sample.** Draw $n$ row indices from $\{0,\dots,n-1\}$ **with replacement**. A row is missing from a given draw with probability $(1-1/n)^n\to e^{-1}\approx 0.368$. Those missing rows are the tree's **out-of-bag (OOB)** rows. The counts of each row form the in-bag weights.
- **Feature subsampling.** At each split, a tree considers a random subset of the $d$ features (often about $\sqrt d$ for classification). This is what separates a random forest from plain bagging.
- **OOB error.** Predict each training row with only the trees whose bootstrap sample excluded it. The result estimates test error without a holdout set.
- **Two importance measures:**
  - **MDI** (mean decrease in impurity): the impurity decrease weighted by the fraction of training samples reaching each node, summed over splits on a feature and averaged over trees (usually normalised). Computed on training data.
  - **Permutation importance:** shuffle one column of held-out data and record the drop in the score. Its mean and spread come from repeated shuffles.

## Worked example

Take $n=4$ rows, two trees and seed 0. Each tree draws `rng.integers(0, 4, size=4)` from one generator, in tree order. Tree 0 draws rows $(3,2,2,1)$, so its counts are $(0,1,2,1)$. The reference protocol of the problem gives this table:

| tree | in-bag counts (rows 0–3) | OOB rows |
|---|---|---|
| 0 | $0,1,2,1$ | row 0 |
| 1 | $3,1,0,0$ | rows 2, 3 |

Every row of counts sums to $n=4$. Over $n=2000$ rows and 20 trees with seed 0, the mean OOB fraction is $0.368$, against $e^{-1}\approx 0.3679$.

**Impurity versus held-out importance (simulated).** The label is a fair coin, independent of every feature. From one `rng = default_rng(0)`, draw in order `y = rng.integers(0, 2, 2000)`, a binary column with the same call, then a continuous column with `rng.normal(size=2000)`. Setup: $n=2000$, a 50/50 split with `random_state=0`, `RandomForestClassifier(n_estimators=200, random_state=0)`, permutation importance on the held-out half with `n_repeats=10`, `random_state=0`, accuracy.

| feature | MDI (training) | permutation drop (held-out) |
|---|---|---|
| binary noise | 0.0031 | 0.0143 (sd 0.0094) |
| continuous noise | 0.9969 | 0.0181 (sd 0.0118) |

Held-out accuracy is 0.511, consistent with chance on these independent noise features. MDI still ranks the continuous noise feature first.

## Why it works

**Variance of an average.** Let each tree have variance $\sigma^2$, and let any two trees have correlation $\rho$. The variance of the average of $B$ trees is
$$\operatorname{Var}=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2.$$
Adding trees shrinks only the second term. Bagging makes the trees different, and feature subsampling lowers $\rho$, which shrinks the first term. This is Breiman's (2001) argument.

**Why OOB is an honest check.** A row's OOB trees never saw it, so their vote is a fair prediction. Averaging those votes over all rows gives an error estimate with no extra holdout.

**Why permutation importance is measured out of sample.** Shuffling a column on held-out data breaks its link to the target, and the drop shows how much the model depends on it. On training data the same shuffle also measures how much the model memorised.

## Signal vs. noise

- **Signal:** bagging plus feature subsampling is the variance reduction. The OOB estimate is free validation.
- **Signal:** an unused feature has exactly zero permutation drop. The converse fails: a used feature can have zero drop because of redundancy, an insensitive metric, or sampling noise.
- **Noise for now:** the exact number of trees.

## Where it breaks

- **MDI favours continuous and many-valued features.** A continuous column offers many cut points, so some look useful by chance. In the simulation above, pure noise gets 0.9969 of the importance.
- **Correlated features share credit.** Setup: $x\sim U(-1,1)$, $y=\mathbf 1[x+\varepsilon>0]$ with $\varepsilon\sim N(0,0.3^2)$, $n=2000$, the same split and forest as above. Reset `default_rng(0)`, draw `x` first and the Gaussian noise second. With one copy of $x$, held-out permutation importance is 0.31. With two identical copies it is 0.18 and 0.26 (each copy's drop falls because the other still carries the signal). MDI splits the total 1.0 into 0.47 and 0.53.
- **A real correlated pair can hide its joint signal.** In `breast_cancer`, `mean radius` and `mean perimeter` correlate at 0.9979. A standardized logistic model gets held-out AUC 0.9251. Permuting the columns individually drops AUC by 0.0969 and 0.1461; permuting them together drops it by 0.4260. The other correlated column partly carries the signal when only one is shuffled, so individual permutation drops measure reliance conditional on the other columns, not the pair's total value.

- **Permutation importance on correlated inputs** can ask the model about unlikely combinations of features. The drop then mixes importance with extrapolation.
- **OOB is optimistic** when rows are dependent, as in time series.

## Interview angle

*60-second answer:* A random forest averages trees fitted on bootstrap samples and random feature subsets. Bagging and feature subsampling decorrelate the trees, which lowers the variance of the average. The out-of-bag rows give a validation estimate for free. For importance, I trust permutation importance measured on held-out data more than MDI, because MDI is biased toward continuous and high-cardinality features. When features are correlated, the importance is shared between them, so I check groups of correlated columns together.

*Likely follow-ups:* Why does feature subsampling help? (It lowers the correlation between trees, which is the dominant term in the variance.) Why is OOB optimistic on time series? (Bootstrap rows are not independent, so an OOB row has near neighbours in its tree's training set.)

## Connections

- **Prerequisites:** `cml/choosing-a-tree-split` (the trees), `cml/bootstrap-and-permutation-resampling` (resampling and permutation tests).
- **Unlocks:** `cml/adaboost-and-gradient-boosting` (boosting, which reduces bias where bagging reduces variance), `cml/bagging` (the general method).
- **Lineage:** Breiman (2001) introduced random forests. ESL chapter 15 covers the variance argument, OOB error and importance measures.
- **References:** [Breiman (2001)](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) develops random forests; the [permutation-importance guide](https://scikit-learn.org/stable/modules/permutation_importance.html) explains correlated-feature limitations.
