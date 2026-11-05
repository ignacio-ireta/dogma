---
title: ROC, PR curves and AUC
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Each sample has a label $y\in\{0,1\}$ and a score $s$. A threshold $t$ flags every sample with $s\ge t$. Sweep $t$ from high to low, and record the counts at each step. With $P$ positives and $N$ negatives:

| | ROC curve | Precision-recall curve |
|---|---|---|
| Horizontal axis | $\mathrm{FPR}=\mathrm{FP}/N$ | recall $=\mathrm{TP}/P$ |
| Vertical axis | $\mathrm{TPR}=\mathrm{TP}/P$ | precision $=\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$ |
| Random ranking | diagonal, area $0.5$ | flat line at the prevalence $\pi=P/(P+N)$ |
| Depends on class ratio? | no | yes |

Two summaries condense the curves:

- **ROC AUC** $=P(s_{\text{pos}}>s_{\text{neg}})+\tfrac12 P(s_{\text{pos}}=s_{\text{neg}})$, the probability that a random positive outranks a random negative, with a tie counting as one half.
- **Average precision (AP)** $=\sum_k (R_k-R_{k-1})\,P_k$ over the distinct thresholds, highest first, with $R_0=0$. Here $R_k$ and $P_k$ are recall and precision at threshold $k$. Each threshold adds the recall it gains, weighted by the precision there. There is no interpolation between points.

## Worked example

Four samples: $y=(0,0,1,1)$ and $s=(0.1,0.4,0.35,0.8)$.

**AUC.** The positives are $0.35$ and $0.8$. The negatives are $0.1$ and $0.4$. Of the four pairs, $0.35>0.1$, $0.35<0.4$, $0.8>0.1$ and $0.8>0.4$ hold, so ordered-correctly pairs are 3 of 4. AUC $=0.75$.

**AP.** Sweep the thresholds from the top:

| Threshold | TP | FP | $P_k$ | $R_k$ | Step $(R_k-R_{k-1})P_k$ |
|---|---|---|---|---|---|
| 0.80 | 1 | 0 | 1 | 0.5 | $0.5\cdot1=0.5$ |
| 0.40 | 1 | 1 | 0.5 | 0.5 | $0$ |
| 0.35 | 2 | 1 | 2/3 | 1 | $0.5\cdot 2/3=1/3$ |
| 0.10 | 2 | 2 | 0.5 | 1 | $0$ |

AP $=5/6=0.833$.

**Imbalance.** Take 100 positives and 9,900 negatives. A threshold with FPR $1\%$ flags 99 negatives. With TPR $80\%$ it also flags 80 positives. Precision is $80/179=0.447$. The ROC point $(1\%, 80\%)$ looks excellent, and the precision point is 45%.

## Why it works

**Ranks give AUC.** The Mann-Whitney statistic counts the ordered pairs, with ties as halves. Its normalised form is $\mathrm{AUC}=(R_{\text{pos}}-P(P+1)/2)/(PN)$, where $R_{\text{pos}}$ is the sum of the positives' ranks. Sorting makes it $O(n\log n)$, instead of the $O(n^2)$ of counting pairs one by one.

**Why ROC looks optimistic.** TPR and FPR are rates. FPR divides by the large negative count $N$, so 99 false positives look like a small number. Under a pure prevalence change that preserves class-conditional score distributions, ROC stays fixed. Precision divides by the number flagged and moves with the base rate. Give each negative 100 copies (99 extra) in the four-sample example. AUC stays at $0.75$, because it averages over negatives. AP falls to $\tfrac12+\tfrac{1}{102}\approx0.510$: the lower positive now enters after 100 higher-scoring negative copies. Saito and Rehmsmeier (2015) make the general case for precision-recall plots under imbalance, and Davis and Goadrich (2006) relate the two curves.

**Why AP has no interpolation.** Between two PR points, precision is not linear in recall, so connecting them with a straight line gives a different area. On the example, the trapezoid rule over the same PR points, starting at recall 0 with precision 1 (what `sklearn.metrics.auc` gives on the precision-recall curve), gives $0.792$, not $0.833$. The step sum is the definition.

## Signal vs. noise

- **Signal:** AUC is a ranking statistic. It compares models on one dataset, and it cannot say whether a threshold is usable.
- **Signal:** at a deployed threshold, the numbers that matter are precision and recall at that threshold.
- **Signal:** a random ranking has AP near the prevalence $\pi$, because precision at any threshold stays near $\pi$. Report AP next to $\pi$, and compare with the 0.5 baseline for AUC.
- **Noise for now:** arguing whether ROC or PR always "wins". Even on balanced data they summarize different aspects of ranking; choose the view that matches the decision.

## Where it breaks

- **Heavy imbalance.** The worked example above has a 1% false positive rate and 80% recall, so its ROC point looks excellent, yet precision is 45%. A high ROC-AUC does not guarantee usable precision at your threshold.
- **Calibration.** AUC is blind to calibration. Scores of $0.01$ for negatives and $0.02$ for positives give AUC $1.0$, but those numbers are useless as probabilities.
- **Ties.** A model with a few distinct scores has many tied pairs. Ties must count as halves, and AP must treat each tie group as one threshold.
- **Single class.** AUC needs both classes. This course's AP requires at least one positive; all-positive data gives AP $1$, while no-positive data raises `ValueError`. Scikit-learn instead reports $0$ with a warning for no positives. State the convention rather than silently inventing $0.5$.

## Interview angle

"The ROC-AUC is 0.95 on a fraud problem with 1% positives. Is the model useful?" Answer shape: AUC is a rank statistic that ignores prevalence. Compare AP with the 1% baseline, and look at precision and recall at the threshold you would deploy. Follow-ups are in `interview.md`.

## Connections

- **Prerequisites:** `cml/precision-recall-and-f1` (every point on these curves is a precision-recall pair).
- **Unlocks:** `cml/handling-class-imbalance` (metric and threshold choice when positives are rare), `cml/probability-calibration` (what AUC cannot check).
- **Lineage:** Mann and Whitney's 1947 rank test gave the statistic. Hanley and McNeil (1982) tied the area under the ROC curve to this Wilcoxon (Mann-Whitney) statistic, and Fawcett (2006) set out ROC analysis for classifiers.
