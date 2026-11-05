---
title: Nearest-neighbour search and KNN
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A KNN model stores the training pairs $(\mathbf{x}_j, y_j)$, $j = 1, \dots, n$, and does no fitting. For a query $\mathbf{q}$, compute the Euclidean distances $d_j = \lVert \mathbf{x}_j - \mathbf{q} \rVert_2$. Take the $k$ rows with the smallest distance, with equal distances broken by lower index. Then:

- **Classification:** the most frequent label among the $k$ neighbours.
- **Regression:** the mean of their targets, or the average weighted by $1/d_j$.

A vote tie goes to the label of the nearest tied neighbour. A zero distance under inverse-distance weights makes the weight infinite, so those exact matches decide alone.

Each query costs $O(nd)$ distance work plus $O(n\log n)$ for the full sort used here, with brute force, for $n$ rows of $d$ features. The training cost is zero. That is why KNN is called lazy learning.

## Worked example

Training points $(0,0)$, $(2,0)$, $(1,0)$, $(9,9)$ have labels $7, 3, 5, 9$. Query $(1.2, 0)$, $k = 2$. Distances are $0.2$ for label 5 and $0.8$ for label 3, so the vote is a tie, and the nearest tied neighbour has label 5. A rule that picks the smallest label would answer 3.

| Method | Neighbours | Prediction |
|---|---|---|
| Uniform mean, $k=2$ | $y = 5, 3$ | $4.0$ |
| Inverse-distance, $k=2$ | weights $5, 1.25$ | $(5\cdot5 + 1.25\cdot3)/6.25 = 4.6$ |

Scaling changes the neighbours. Take query $(50000, 30)$, $P_1 = (50300, 31)$, $P_2 = (50005, 60)$. Raw Euclidean distance picks $P_2$ ($30.4$ against $300.0$). If you divide by feature standard deviations $20000$ and $10$, the distances to $P_1$ and $P_2$ become $0.101$ and $3.00$, respectively, and $P_1$ is nearest.

Dimension changes what "near" means. For each dimension, reset `default_rng(0)`, draw 1000 uniform points in $[0,1]^d$, then 50 queries from the same distribution. The mean farthest-to-nearest distance ratio is $81.3$ at $d = 2$, $3.92$ at $d = 10$, $1.44$ at $d = 100$ and $1.12$ at $d = 1000$.

## Why it works

**Locality.** KNN assumes that nearby inputs have similar outputs. The classic analysis is Cover & Hart (1967). As $n$ grows, the one-nearest-neighbour error is at most twice the Bayes error rate in the two-class case. See also Hastie, Tibshirani & Friedman, nearest-neighbour chapter.

**$k$ is a bias-variance knob.** With $k = 1$ on distinct points, every training point is its own neighbour. Reset `rng = default_rng(0)`, draw `X = rng.normal(size=(200, 2))`, then `y = rng.integers(0, 2, 200)`. Training accuracy is $1.000$. With $k = 15$ it falls to $0.585$, and with $k = n = 200$ the model predicts the majority label for everything, giving $0.540$. Small $k$ fits noise, and large $k$ smooths the boundary away.

**Scaling is part of the model.** The metric is the model. Standardise on training data only, then apply the same scaler to queries. See `cml/feature-scaling`.

## Signal vs. noise

- **Signal:** the algorithm is a sort by distance, then a vote or an average. The tie rules are part of the contract.
- **Signal:** scaling decides the neighbours, and $k$ decides the bias-variance trade-off.
- **Noise for now:** the exact index structure. Brute force is correct and is the reference you test against.

## Where it breaks

- **Unscaled features.** One feature with a large range dominates every distance.
- **High dimension.** Distance concentration: the nearest and farthest points become similar, so "nearest" carries little information. The ratio above drops from $81$ to $1.1$.
- **Large $n$ at query time.** Brute force costs $O(nd)$ per query. Each query scans every row.
- **Class imbalance.** A majority class wins most votes near a boundary. Re-balance the training set, and choose $k$ on a metric that reflects the cost of errors.

## Interview angle

*60-second answer:* KNN stores the data, and a prediction is a vote or an average over the $k$ nearest rows under a metric. It has no training step, and the metric is the model, so features must be scaled. $k$ trades bias for variance. In high dimension, distance concentration weakens the signal, so the choice of features and the embedding matter more than the algorithm. For large search, use an approximate index, such as a graph (HNSW) or an inverted file, and measure recall against brute force.

*Likely follow-ups:* How do you search a million embeddings quickly? (Approximate nearest-neighbour index, tuned for recall against brute-force ground truth.) How do k-d trees differ? (They are exact, prune with axis-aligned bounds, and stop helping as dimension grows.)

## Connections

- **Prerequisites:** `cml/vector-operations` (distances and dot products), `cml/feature-scaling` (the metric).
- **Unlocks:** retrieval for RAG and vector databases, which are approximate KNN over embeddings, and the clustering topics, which use the same distance.
- **Lineage:** Cover & Hart 1967 for the classifier. Malkov & Yashunin 2016 for HNSW, the graph index that most vector stores use.
- **References:** [Cover & Hart (1967)](https://isl.stanford.edu/~cover/papers/transIT/0021cove.pdf) proves the asymptotic nearest-neighbour error bound.
