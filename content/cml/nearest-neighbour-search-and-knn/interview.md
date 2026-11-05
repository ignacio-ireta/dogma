# Interview notes — Nearest-neighbour search and KNN

## Q1. Walk me through a KNN classifier, including ties and scaling.

**60-second answer.** KNN stores the training set. For a query, it computes the distance to every training row, takes the $k$ nearest with equal distances broken by lower index, and votes. A vote tie goes to the nearest tied neighbour. Regression averages the targets, optionally weighted by $1/d$. Because the metric is the model, features must be standardised on the training data. Training costs nothing, and distance computation costs $O(nd)$ per query, with $O(n\log n)$ more for a full sort.

**Follow-ups.**
- *Why is a zero distance special?* Its inverse-distance weight is infinite, so the exact matches must decide alone.
- *How do you handle categorical features?* Encode them so that the distance means something, such as one-hot with a weight, or use a different metric. Euclidean distance on arbitrary integer codes is meaningless.

## Q2. How would you find the nearest vectors among a million embeddings quickly?

**60-second answer.** Brute force is a correct baseline, but it scans every row per query. For speed, use an approximate index. Graph indexes such as HNSW walk a layered neighbour graph and return candidates with high recall. Inverted-file indexes cluster the vectors and search only the nearest clusters. You set the speed and recall trade-off with index parameters. You measure recall@k against brute force on a sample of queries, and you report latency at a stated recall.

**Follow-ups.**
- *When would you keep an exact search?* When the data is small, when the metric is unusual, or when recall must be exact, for example for audit.
- *What does the vector database add?* Storage, updates, filtering and serving around the same index.

## Q3. How do you choose $k$, and why does high dimension hurt KNN?

**60-second answer.** Choose $k$ by cross-validation on a metric that matches the cost of errors. Small $k$ has high variance, and large $k$ has high bias. In high dimension, distances concentrate: the farthest and nearest points look alike for typical data, so the neighbour set becomes noisy. The fix is better features, a learned embedding, or dimension reduction, not a smaller $k$.

**Follow-ups.**
- *Is $k$ odd for binary problems?* Odd $k$ avoids vote ties, but the tie rule in this course handles them anyway.
- *What breaks first in practice?* Usually scaling and class imbalance, before the choice of $k$.

## Common mistakes

- Scaling the query and training data separately, or fitting the scaler on the whole dataset. Fit on training data only.
- Using a smallest-label rule for vote ties without saying so. The nearest tied neighbour is the deterministic rule here.
- Reporting an approximate index's speed without its recall.
- Taking the nearest neighbour's distance as a confidence score without calibration.
