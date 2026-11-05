## Hint 1 — concept

A KNN regressor reuses the classifier's neighbour search and replaces the vote with an average. The only new question is how much each neighbour counts, and what to do when a neighbour sits exactly on the query and its weight would be infinite.

## Hint 2 — formula / approach

Find the $k$ nearest rows exactly as in the classifier. For `"uniform"`, return the mean of their targets. For `"distance"`, use $w_j = 1/d_j$ and return $\sum_j w_j y_j / \sum_j w_j$. If any $d_j = 0$ among the $k$ neighbours, return the mean of the targets at distance $0$ instead.

## Hint 3 — code shape

Validate first: `weights` is one of the two strings, `X_train` and `X_query` are finite 2-D arrays with matching columns, `y_train` is a finite 1-D array of the right length, and `k` is an integer in $[1, n]$. Then, for each query row, compute distances, take the first `k` of `np.lexsort((np.arange(n), dist))`, and branch on `weights`. In the distance branch, test `d == 0` before dividing.
