---
title: K-means
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

K-means partitions $n$ samples $\mathbf{x}_i \in \mathbb{R}^d$ into $k$ clusters. It minimises the **inertia**, the within-cluster sum of squared distances to the centroids $\mathbf{c}_1, \dots, \mathbf{c}_k$:

$$J = \sum_{i=1}^{n} \lVert \mathbf{x}_i - \mathbf{c}_{\ell_i} \rVert_2^2,$$

where $\ell_i \in \{1, \dots, k\}$ is the label of sample $i$. Lloyd's algorithm alternates two steps, each of which minimises $J$ over one set of variables while the other is fixed.

1. **Assign.** Given the centroids, give each sample its nearest centroid, $\ell_i = \arg\min_j \lVert \mathbf{x}_i - \mathbf{c}_j \rVert_2^2$.
2. **Update.** Given the labels, set each centroid to the mean of its samples, $\mathbf{c}_j = \frac{1}{|S_j|}\sum_{i \in S_j} \mathbf{x}_i$, where $S_j = \{i : \ell_i = j\}$.

The mean is the minimiser because setting $\partial/\partial \mathbf{c}_j \sum_{i \in S_j}\lVert \mathbf{x}_i - \mathbf{c}_j\rVert^2 = -2\sum_{i \in S_j}(\mathbf{x}_i - \mathbf{c}_j)$ to zero gives the mean. An empty cluster has no mean. In this course it keeps its previous centroid.

## Worked example

Six points, $k = 2$: $(0,0), (4,0), (0,4), (4,4), (2,9), (2,6)$. The same data settles at different fixed points depending on the start:

| Start centroids | Final centroids | Inertia | Iterations |
|---|---|---|---|
| $(0,0)$, $(0,4)$ | $(2, 0)$, $(2, 5.75)$ | $32.75$ | 2 |
| $(0,4)$, $(2,9)$ | $(2, 2)$, $(2, 7.5)$ | $36.50$ | 3 |
| $(0,0)$, $(4,0)$ | $(1.33, 6.33)$, $(2.67, 1.33)$ | $36.67$ | 3 |

Each row is a fixed point: another step would change nothing. The algorithm cannot tell from inside a run which fixed point is best.

The inertia falls at every step of a run. On the 1-D points $0, 1, 2, 10, 11, 12$ with starts $0$ and $1$, the inertia after each iteration is $50.32$, then $4.00$, then $4.00$.

## Why it works

**Monotone decrease.** The assignment step can only lower each sample's squared distance, so $J$ does not rise. The update step moves each centroid to the minimiser of its own cluster's term, so $J$ does not rise either. $J \ge 0$ and there are finitely many partitions, so the sequence converges. It converges to a local minimum, and finding the global minimum is NP-hard in general.

**Stopping.** The test is the largest centroid shift, $\max_j \lVert \mathbf{c}_j^{(t)} - \mathbf{c}_j^{(t-1)} \rVert_2 \le \texttt{tol}$. A shift of exactly `tol` counts as converged.

**k-means++ seeding.** Choose the first centre uniformly. Choose each next centre with probability proportional to its squared distance $D^2$ to the nearest centre so far. Arthur & Vassilvitskii (2007) show that the expected cost is within an $O(\log k)$ factor of the optimum. For points $0, 1, 3$ with the first centre at $0$, $D^2 = (0, 1, 9)$, so the next centre is $3$ with probability $0.9$ and $1$ with probability $0.1$. Averaged over a uniform first centre, the point $3$ is chosen with probability $0.567$.

## Signal vs. noise

- **Signal:** the two steps, and the fact that each lowers $J$. That is the whole guarantee.
- **Signal:** the result depends on the start. Use k-means++ and several restarts, and keep the lowest inertia.
- **Noise for now:** library defaults such as how empty clusters are handled. Know your rule, and state it.

## Where it breaks

- **Non-spherical clusters.** Each cluster is a Voronoi cell, so boundaries are straight bisectors. Concentric rings get split across the middle.
- **Unscaled features.** Distances are dominated by the feature with the largest range. Standardise first, as in `cml/feature-scaling`.
- **Outliers.** A mean moves toward a distant point, so one outlier can drag a whole centroid.
- **Choosing $k$.** Inertia always falls as $k$ grows, so the elbow is a judgement. Use it with a silhouette or a downstream check, as in `cml/cluster-quality-scores`.

## Interview angle

*60-second answer:* K-means alternates nearest-centroid assignment and mean updates. Each step lowers the inertia, so it converges, but to a local minimum. The start matters, so use k-means++ seeding and several restarts. Choose $k$ with an elbow or a silhouette, not inertia alone. K-means assumes roughly spherical clusters on comparable scales, and it is sensitive to outliers. Scale the features first.

*Likely follow-ups:* Why is the centroid the mean? (It minimises the squared error of its cluster, which is the derivative-zero argument.) When would you use something else? (A GMM when clusters have different shapes and soft membership; DBSCAN when the clusters are arbitrary-shaped and noise matters.)

## Connections

- **Prerequisites:** `cml/nearest-neighbour-search-and-knn` (the nearest-centroid rule), `cml/feature-scaling` (the metric).
- **Unlocks:** cluster quality scores, hierarchical clustering and Gaussian mixtures with EM, where k-means is the hard-assignment limit.
- **Lineage:** Lloyd's algorithm, then k-means++ by Arthur & Vassilvitskii (2007). Bishop's chapter on mixture models and EM covers the GMM link.
- **References:** [The clustering guide](https://scikit-learn.org/stable/modules/clustering.html#k-means) describes Lloyd updates and links the original k-means++ paper.
