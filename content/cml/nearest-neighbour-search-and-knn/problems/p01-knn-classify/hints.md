## Hint 1 — concept

A KNN classifier has no training step. A prediction is a vote among the training rows closest to the query. Two things need care: the order of the neighbours, which decides which rows are in the vote, and the rule for a tie in the vote itself.

## Hint 2 — formula / approach

For a query $\mathbf{q}$, compute $d_j = \lVert \mathbf{x}_j - \mathbf{q}\rVert_2$ for every training row $j$. Sort the rows by the pair $(d_j, j)$, so equal distances are ordered by index. Keep the first $k$. Count their labels. If the top count is shared, walk the $k$ neighbours in that same order and return the first label whose count is at the top.

## Hint 3 — code shape

Validate first: `X_train` and `X_query` are finite 2-D arrays with the same number of columns, `y_train` is a 1-D integer array of the right length, and `k` is an integer with $1 \le k \le n$. Then loop over the query rows. For each one, compute the distances, sort the indices by (distance, index), slice the first `k`, and count the labels with `collections.Counter` or `np.unique`. Finally, pick the tied label by scanning the sorted neighbours.
