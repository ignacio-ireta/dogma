---
title: Confusion matrix and count-based scores
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Take one binary classifier at one threshold. Each sample has a true label $y\in\{0,1\}$ and a predicted label $\hat y\in\{0,1\}$. Counting the four pairs gives the **confusion matrix**. Every score in this topic is a function of these four counts.

| | predicted 1 | predicted 0 |
|---|---|---|
| actual 1 | TP | FN |
| actual 0 | FP | TN |

Rows are true labels and columns are predicted labels, the scikit-learn convention. Transposing the matrix raises no error: it exchanges FP and FN, so precision and recall exchange roles. Accuracy, F1 and MCC stay unchanged.

Let $N=\mathrm{TP}+\mathrm{FP}+\mathrm{FN}+\mathrm{TN}$.

| Score | Formula | Question it answers |
|---|---|---|
| Accuracy | $(\mathrm{TP}+\mathrm{TN})/N$ | What share of all samples is right? |
| Precision | $\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$ | Of the flagged samples, how many are real? |
| Recall (TPR, sensitivity) | $\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$ | Of the real positives, how many are flagged? |
| Specificity (TNR) | $\mathrm{TN}/(\mathrm{TN}+\mathrm{FP})$ | Of the real negatives, how many are cleared? |
| F1 | $2\mathrm{TP}/(2\mathrm{TP}+\mathrm{FP}+\mathrm{FN})$ | One number that balances precision and recall |
| MCC | $\dfrac{\mathrm{TP}\,\mathrm{TN}-\mathrm{FP}\,\mathrm{FN}}{\sqrt{(\mathrm{TP}+\mathrm{FP})(\mathrm{TP}+\mathrm{FN})(\mathrm{TN}+\mathrm{FP})(\mathrm{TN}+\mathrm{FN})}}$ | Balanced correlation of predictions and labels, in $[-1,1]$ |

**Zero denominators.** A ratio such as precision is undefined when nothing is predicted positive. This course reports it as $0.0$, the scikit-learn convention `zero_division=0`. The choice must be stated. NaN is never returned, because one NaN silently poisons every average and comparison that uses it.

## Worked example

Fraud screening: 1,000 transactions, 10 of them fraudulent.

**Model A** flags nothing: TP=0, FP=0, FN=10, TN=990.
- Accuracy $990/1000 = 0.99$. Recall $0/10 = 0$. Specificity $990/990 = 1$.
- Precision has no flagged samples, so it is $0/0$, reported as $0.0$. F1 is $0$. MCC is $0$, because $\mathrm{TP}+\mathrm{FP}=0$.

**Model B** flags 48: TP=8, FP=40, FN=2, TN=950.
- Accuracy $958/1000 = 0.958$. Precision $8/48 = 0.167$. Recall $8/10 = 0.8$. Specificity $950/990 = 0.960$.
- F1 $16/58 = 0.276$. MCC $= (8\cdot950 - 40\cdot2)/\sqrt{48\cdot10\cdot990\cdot952} = 0.354$.

Model A has the higher accuracy, and it is worthless: it never finds fraud. Model B is worse on accuracy and far more useful. MCC gives A the value of "no information" (0) and gives B a positive score.

## Why it works

Each score conditions on a different part of the table. Precision conditions on the prediction ("given flagged"). Recall conditions on the truth ("given fraud"). Specificity is the recall of the negative class. Because they condition on different things, they can disagree sharply. Model A has specificity 1 and recall 0 at the same time.

MCC is the phi coefficient: the Pearson correlation between the two binary vectors, predicted and true (Matthews 1975). If either vector is constant, correlation is undefined and we report $0$ by convention. With both classes present, perfect and fully inverted predictions give $1$ and $-1$. It uses all four cells symmetrically, so swapping which class is "positive" leaves accuracy and MCC unchanged, while precision and recall can change.

## Signal vs. noise

- **Signal:** accuracy means nothing without the base rate. Report the prevalence next to it. With 1% positives, an all-negative model scores 99%.
- **Signal:** the table is enough. These scores need no raw scores, only the counts at one threshold.
- **Signal:** precision and specificity look similar in name but condition on different sets. Precision uses predicted positives, specificity uses actual negatives.
- **Noise for now:** the long list of aliases (hit rate, sensitivity, true positive rate). Learn one name and map the rest to it.

## Where it breaks

- **Imbalance.** Accuracy and specificity look excellent when negatives dominate. Check recall and precision for the class you care about.
- **Transposed or flipped matrix.** Rows and columns swapped, or the positive class changed, still gives plausible-looking numbers.
- **Zero denominators.** A model that never predicts positive has undefined precision. Return $0.0$ by an explicit convention, and say so.
- **Threshold dependence.** The table belongs to one threshold. Compare thresholds on the same evaluation samples, reporting each threshold with its counts and scores.

## Interview angle

"The model is 99% accurate. Is it good?" Answer shape: not without the base rate. If positives are 1%, the all-negative model also scores 99%. Report recall and precision for the positive class, and MCC as one balanced number. Follow-ups are in `interview.md`.

## Connections

- **Prerequisites:** `cml/probability-basics`. Precision is $P(y=1\mid\hat y=1)$ and recall is $P(\hat y=1\mid y=1)$, so the conditioning in the table is the conditioning from that topic.
- **Unlocks:** `cml/precision-recall-and-f1` (trading precision against recall), `cml/roc-pr-curves-and-auc` (sweeping the threshold), and `cml/handling-class-imbalance`.
- **Lineage:** Matthews (1975) proposed the coefficient in a structural biology paper that compared predicted and observed protein structure.
