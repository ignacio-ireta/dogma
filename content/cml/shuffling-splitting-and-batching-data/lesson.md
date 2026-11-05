---
title: Shuffling, splitting and batching data
minutes: 12
min_viable: "Read The system and Where it breaks, then solve p01."
---

## The system

A random split assumes the rows are **i.i.d.**: independent draws from one distribution. Under that
assumption, a random subset is a fair stand-in for the whole, and shuffling removes any order the
file happened to have. Three rules follow.

1. **Count rule.** The test side gets $n_{\text{test}} = \lceil t\cdot n\rceil$ rows, where $t$ is the
   test fraction. Use the ceiling: $\lceil 2.5\rceil=3$, while round-to-even gives $2$. The product is
   float64, so $0.07\cdot100=7.000000000000001$ and the ceiling gives $8$, not $7$. The protocol keeps
   the float, as scikit-learn does.
2. **Stratify when classes are uneven.** Split each class separately, so each class keeps its share
   on both sides. The class $j$ ideal is $t\,n_j$. Give each class $\lfloor t\,n_j\rfloor$ rows, then
   hand the leftover rows to the classes with the largest fractional parts.
3. **One generator, one draw per permutation.** The seed fixes every order. Rebuilding the generator
   inside a loop repeats the same shuffle.

Batching is the same protocol applied per epoch: draw one permutation, cut it into slices of $B$ rows,
and keep or drop the short last slice.

## Worked example

Take $n=10$ and $t=0.25$. The ideal test count is $2.5$, so $n_{\text{test}}=3$.

Now stratify $n=10$ rows with labels $y = (0,0,0,0,1,1,1,1,1,1)$, so $n_0=4$ and $n_1=6$, at $t=0.5$.
The ideals are $2.0$ and $3.0$. Both are whole numbers, so $R = 5 - (2+3) = 0$: no leftover rows.
The test side takes $2$ rows of class $0$ and $3$ of class $1$. Had the ideals been $2.5$ and $2.5$,
the one leftover row would go to the class listed first, because the fractional parts tie.

## Why it works

Stratification fixes each class count close to its requested share. Integer rounding can change the
share slightly, but the count no longer fluctuates across random permutations.
Take $200$ rows with $10$ positives, and a $20\%$ test split of $40$ rows. The number of positives in
a random test set is hypergeometric. The chance it has **no** positive at all is
$\binom{190}{40}/\binom{200}{40} \approx 0.10$. A stratified split always gives exactly $2$ positives.
That gap explains why stratification helps rare-class evaluation. It cannot create positive rows
when there are too few, and group or time constraints may take priority. See
ESL, chapter 7 (model assessment), gives the setting.

The i.i.d. assumption is what licenses the random split. It is a property of how the data was
generated, not of the file, so it must be checked before the split is chosen.

## Signal vs. noise

- **Signal:** the ceiling rule, one generator per call, per-class draws in ascending class order, and
  a split decided before any model is fitted.
- **Signal:** a class with one row goes to training unless the largest-remainder rule gives it a test
  share. In that case the protocol raises, because the class would have no training row.
- **Noise for now:** how one library breaks ties between equal fractional parts. Any rule that is
  stated and fixed is a valid protocol.

## Where it breaks

- **Time.** Forecasting asks for future performance from past training data. A random split puts
  future rows in training and past rows in the test set. Split by time instead.
- **Groups.** Several rows from one patient or user are correlated. A random split can put the same
  entity on both sides, so the model is scored on people it has already seen. Split by group.
- **Duplicates.** A repeated row on both sides inflates the test score in the same way.
- **Sorted batches.** Mini-batches cut from sorted data contain one class or one time range each.
  Reshuffle each epoch, as covered in the stochastic gradient lesson.
- **Tiny classes.** A class of one or two rows cannot be split usefully. Merge it or collect more data.

## Interview angle

Start with the structure of the rows, not the algorithm. Random splitting is right for i.i.d. rows.
Stratify when a class is rare, so each side keeps its share. Use a time split for temporal data, and a
group split when rows share an entity. Say which of these you chose and why, because a leak in the
split shows up as a score that is too good to be true.

Follow-ups: how many repeats a split needs (repeat the whole protocol with several seeds and report
the spread of the metric). What to do with a rare class (stratify, then check the class count on each
side).

## Connections

- **Prerequisites:** `cml/stochastic-and-mini-batch-gradient-descent` (the epoch and the reshuffling
  rule).
- **Unlocks:** `cml/feature-scaling` and `cml/preprocessing-pipelines` (fitted statistics must come
  from the training side only), `cml/k-fold-cross-validation` (repeated splits), and
  `cml/leakage-and-walk-forward-validation` (what a time split must look like).
