---
title: Precision, recall and F1
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

For one class, let TP, FP and FN be the counts of true positives, false positives and false negatives.
**Precision** $P=\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$ is the share of flagged samples that are real. **Recall** $R=\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$ is the share of real samples that are flagged.

The two move against each other through the threshold $t$. Flagging a sample when its score is at least $t$ means lowering $t$ flags more samples. Recall then rises, and precision usually falls. Raising $t$ does the reverse. The curve is not strictly monotone, so it can wobble.

The F-measure combines the two with a weight $\beta$:

$$F_\beta=\frac{(1+\beta^2)\,P\,R}{\beta^2 P+R}=\frac{(1+\beta^2)\,\mathrm{TP}}{(1+\beta^2)\,\mathrm{TP}+\beta^2\,\mathrm{FN}+\mathrm{FP}}.$$

The parameter $\beta>0$ controls the balance: $\beta>1$ emphasizes recall, with FN weighted by $\beta^2$ relative to FP in the count formula; $\beta<1$ emphasizes precision. $F_1=2PR/(P+R)=2\mathrm{TP}/(2\mathrm{TP}+\mathrm{FP}+\mathrm{FN})$, using zero for a zero denominator (van Rijsbergen, *Information Retrieval*, 1979, evaluation chapter).

For $k$ classes, the per-class values are averaged in one of three ways:

| Average | How | Use when |
|---|---|---|
| Micro | Pool TP, FP, FN over classes, then divide once | each class should count by its samples |
| Macro | Unweighted mean of the per-class values | each class matters equally, including rare ones |
| Weighted | Mean weighted by each class's true count | per-class scores, but mirroring class frequency |

## Worked example

Labels and predictions for 12 samples, with positive class 1:

- $y=(1,0,1,1,0,0,1,0,1,0,0,1)$
- $\hat y=(1,1,0,1,0,0,1,0,1,1,0,1)$

Counting: TP = 5, FP = 2, FN = 1. So $P = 5/7 = 0.714$, $R = 5/6 = 0.833$, and $F_1 = 10/13 = 0.769$.

Multi-class with $y=(0,0,0,0,0,0,1,1,2,2)$ and $\hat y=(0,0,0,1,1,0,1,2,2,0)$:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 | 4/5 | 4/6 | 8/11 |
| 1 | 1/3 | 1/2 | 2/5 |
| 2 | 1/2 | 1/2 | 1/2 |

Macro F1 is the mean of the three F1 values, $(8/11+2/5+1/2)/3 = 179/330 = 0.542$. Micro gives $0.6$ for all three measures, which equals accuracy: 6 of 10 correct. Weighted precision is $0.647$.

## Why it works

**Why a harmonic mean.** Averaging $P=1$ and $R=0.01$ arithmetically gives $0.505$. The harmonic mean gives $0.020$. The harmonic mean is pulled toward the smaller value, and it is zero when either value is zero. A model that flags everything has $R=1$ and tiny $P$, and the harmonic mean refuses to call that good.

**Why F1 is not the macro F1 of the means.** The macro F1 is the mean of the per-class F1 values. Applying the harmonic mean to macro precision and macro recall gives $0.550$ on the example above, not $0.542$. The two numbers are different quantities, and the average over classes must be taken after the per-class step.

**Business framing.** Choose $\beta$, and the threshold, from the cost of each error:

- *Fraud review.* A false positive costs investigator time, and a false negative costs the loss. A review budget caps the number flagged; a precision floor separately limits the share of wasted reviews. Maximise recall subject to the actual constraints.
- *Screening.* A missed case is costly, and a false alarm costs a cheap follow-up test. Favour recall, so $\beta>1$.
- *Retention offers.* A false positive is a wasted offer. Precision matters, and recall matters only up to the value of the customers saved.

## Signal vs. noise

- **Signal:** the threshold belongs to the model's output. Report it with the score, or the number has no meaning.
- **Signal:** micro F1 equals accuracy for single-label multi-class data. If micro looks fine, check the rare classes separately.
- **Signal:** F1 depends on which class is labelled positive. Swapping the two classes can change F1.
- **Noise for now:** the many named F-variants. Learn $F_1$ and the $F_\beta$ form, and derive the rest from the counts.

## Where it breaks

- **Averages hide classes.** Micro is dominated by large classes. Macro gives a tiny class the same weight as a large one, so a noisy class with three samples can swing it.
- **F1 ignores true negatives.** When the negatives are the thing you care about, F1 says nothing about them. Use MCC or specificity alongside it.
- **Tuning on the same data.** The best-F1 threshold found on the training data is optimistic. Choose it on held-out scores.
- **Tied scores.** A threshold cannot separate tied samples, so a sweep that splits ties reports impossible numbers.

**Tolerances merge real ties.** On raw `breast_cancer` (569 rows, 212 malignant), the best F1 of the feature `worst concave points` is $0.886$, at threshold $0.1418$ over 492 distinct candidates. Scale the scores by $10^{-9}$, which keeps their order, and the best F1 is still $0.886$. On those scaled scores, a sweep that merges scores within an absolute tolerance of $10^{-8}$ sees one tie group, so its only candidate flags everything. Its F1 is $2\cdot212/(2\cdot212+357)=0.543$. Tied scores must be grouped by exact equality, or the sweep picks a threshold that splits or merges real ties.

## Interview angle

"Would you optimise precision or recall?" Answer shape: ask what each error costs. Then name the operating point the business can afford, and say why macro and micro can disagree on the same predictions. Follow-ups are in `interview.md`.

## Connections

- **Prerequisites:** `cml/confusion-matrix-and-count-based-scores` (every term here is a ratio of its counts).
- **Unlocks:** `cml/roc-pr-curves-and-auc` (the whole curve over thresholds), `cml/handling-class-imbalance` (what to do when the positive class is rare).
- **Lineage:** van Rijsbergen's *Information Retrieval* (1979) introduced the F-measure to combine precision and recall for retrieval systems.
