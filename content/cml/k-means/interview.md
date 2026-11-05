# Interview notes — K-means

## Q1. Walk me through Lloyd's algorithm, and why does it converge?

**60-second answer.** Given centroids, assign each sample to its nearest centroid. Given the assignments, move each centroid to the mean of its samples. Each step does not increase the inertia, which is the sum of squared distances to the assigned centroids. The inertia is bounded below by zero, and there are finitely many partitions, so the loop stops when the centroids stop moving. It converges to a local minimum. Finding the global minimum is NP-hard in general, so the start matters.

**Follow-ups.**
- *Why is the centroid the mean?* It minimises the sum of squared distances to its cluster, which is the derivative-zero argument.
- *What if a cluster becomes empty?* Its mean is undefined. State the rule. Keeping the previous centroid is one choice, and relocating it to a far point is another.

## Q2. How do you choose the number of clusters?

**60-second answer.** Inertia always falls as $k$ grows, so the minimum inertia is no guide. Look for an elbow in the inertia curve, but treat it as a judgement. Check the silhouette score, the stability across restarts, and a downstream metric. If the clusters feed a decision, the decision's metric is the final judge.

**Follow-ups.**
- *Why not pick the $k$ with lowest inertia?* Because it is $n$, where every sample is its own cluster.
- *How does scaling change the answer?* Unscaled features make one dimension dominate the distances, so the clusters follow that dimension. Standardise first.

## Q3. When would k-means give a wrong answer?

**60-second answer.** K-means assumes roughly spherical clusters of similar size and spread, on comparable scales. Concentric rings and elongated clusters get split across the wrong boundary, because the boundaries are straight bisectors. Outliers pull the means, and one distant point can drag a centroid. Start points that are close together give a poor local minimum. Mitigations are k-means++ with restarts, scaling, robust variants, or a different model, such as a Gaussian mixture for soft, elliptical clusters, or DBSCAN for arbitrary shapes with noise.

**Follow-ups.**
- *Is k-means a special case of something?* Yes. It is the hard-assignment, equal-spherical-variance limit of a Gaussian mixture fitted by EM.
- *Why does k-means++ help?* It spreads the starts, so the first centres fall in different groups, which gives a better expected starting cost.

## Common mistakes

- Reporting the inertia of a single run without restarts. A different start can be far worse.
- Clustering unscaled features, then reading the clusters as if the distance were meaningful.
- Using the mean of an empty cluster, which gives NaN, without a rule for that case.
- Comparing inertia across different $k$ and choosing the smallest.
