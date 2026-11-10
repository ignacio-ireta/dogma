---
title: Ranking metrics
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A retriever returns an ordered list for each query. A ranking metric scores that list against judged items.
Two families exist:

| Metric | Relevance | Position counts? | Denominator | Range |
|---|---|---|---|---|
| Precision at $k$ | binary | only the top $k$ | $k$ | $[0,1]$ |
| Recall at $k$ | binary | only the top $k$ | number of relevant items $\lvert R\rvert$ | $[0,1]$ |
| Reciprocal rank | binary | yes, by the first hit | $1 /$ rank of the first hit | $[0,1]$ |
| Average precision at $k$ (this course convention) | binary | yes, at each hit | $\min(k,\lvert R\rvert)$ | $[0,1]$ |
| NDCG at $k$ | graded | yes, by $\log_2$ discount | ideal DCG | $[0,1]$ |

Gain and discount define NDCG. With relevance $\mathrm{rel}_r$ at 1-based rank $r$:

$$\mathrm{DCG}_k=\sum_{r=1}^{k}\frac{\mathrm{rel}_r}{\log_2(r+1)},\qquad \mathrm{NDCG}_k=\frac{\mathrm{DCG}_k}{\mathrm{IDCG}_k}$$

$\mathrm{IDCG}_k$ is the same sum for the items sorted by relevance, descending. The ratio is 1 for an ideal list.

## Worked example

Graded relevance in ranked order: $\mathrm{rel}=(3,2,3,0,1,2)$. The discounts for ranks 1 to 6 are $1,\ 0.631,\ 0.5,\ 0.431,\ 0.387,\ 0.356$.

- $\mathrm{DCG}_6 = 3+2(0.631)+3(0.5)+0+1(0.387)+2(0.356) = 6.861$.
- Ideal order $(3,3,2,2,1,0)$: $\mathrm{IDCG}_6 = 7.141$.
- $\mathrm{NDCG}_6 = 6.861 / 7.141 = 0.961$.

Binary example: ranked $(d_3,d_1,d_7,d_2)$, relevant $\{d_1,d_2,d_9\}$, $k=3$. The top three hold one hit, at rank 2. Precision@3 $=1/3$, recall@3 $=1/3$, reciprocal rank $=1/2$, AP@3 $=(1/2)/\min(3,3)=0.167$.

## Why it works

**Position matters.** Precision@$k$ treats rank 1 and rank $k$ as equal. The $\log_2$ discount makes early positions weigh more, and graded labels reward a highly relevant item over a marginal one (Järvelin and Kekäläinen 2002).

**The log base cancels.** $1/\log_b(r+1)=\ln b/\ln(r+1)$, so the factor $\ln b$ divides out of the ratio. NDCG gives the same value in base 2, $e$ or 10.

**Ties need a rule.** Equal scores have no defined order. The sklearn default averages over the possible orders, which is what the problem's oracle checks.

**One right answer.** When one item answers the query, as in question answering, the reciprocal rank of that item is the natural score, and the mean over queries is MRR (Voorhees 1999).

**Recall first for RAG.** A generator can only cite what the retriever returned. If the answer is not in the top $k$, no reranking can recover it, so recall@$k$ is the first number to measure.

## Signal vs. noise

- **Signal:** always report the $k$, the number of labelled queries, and the metric. A bare "NDCG 0.82" means nothing.
- **Signal:** match the metric to the task. One right answer: recall@$k$ and MRR. A ranked list with grades: NDCG.
- **Noise:** the difference between two rankers on twenty queries. Use a bootstrap interval before you call it a gain.
- **Noise for now:** exact sklearn behaviour beyond ties. Learn the definitions first.

## Where it breaks

- **Incomplete judgements.** An unjudged relevant item counts as irrelevant, so a better new retriever can look worse.
- **Denominators.** Precision@$k$ divides by $k$ even when the list is shorter. Dividing by the list length inflates it.
- **Offline is not online.** A rise in offline NDCG may not move clicks or answers. Confirm with an A/B test, as in `cml/a-b-test-design-and-analysis`.
- **Leakage.** Tuning on the same queries you report on makes the number optimistic.

**Tolerances merge real score ties.** On raw `breast_cancer` (569 rows), graded relevance from quartiles of `area error` and the score `worst concave points` give NDCG@25 $0.902$. Multiply the scores by $10^{-9}$, which keeps their order, and the value is still $0.902$. A solution that merges scores within an absolute tolerance of $10^{-8}$ sees one tie group, and its NDCG@25 falls to $0.501$, the value of a random order, so the ranking is lost. Tie groups are exact equality.

## Interview angle

"Evaluate this retriever." Answer shape: recall@$k$ over labelled queries first, because the generator is limited by it. Add NDCG@$k$ with graded labels, or MRR when there is one answer, to judge the order. Report $k$, the query count and an interval, then confirm the gain online. Follow-ups are in `interview.md`.

## Connections

- **Prerequisites:** `cml/precision-recall-and-f1` (precision and recall at a cut), `cml/roc-pr-curves-and-auc` (average precision).
- **Unlocks:** `cml/reranking-and-diversity` (NDCG is the usual target), `cml/industrial-ranking-model-components`.
- **Lineage:** Järvelin and Kekäläinen (2002) introduced cumulated gain with a discount. Voorhees (1999) used the reciprocal rank in the TREC-8 question answering track.
- **References:** [Järvelin & Kekäläinen (2002)](https://faculty.cc.gatech.edu/~zha/CS8803WST/dcg.pdf) develops discounted and normalised cumulative gain; [Voorhees (1999)](https://trec.nist.gov/pubs/trec8/papers/qa_report.pdf) defines MRR for TREC QA.
