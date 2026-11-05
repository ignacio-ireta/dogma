## Hint 1 — concept

Lloyd's algorithm alternates two steps. Assigning each sample to its nearest centroid can only lower the squared distances. Moving each centroid to the mean of its samples is the best position for that set of samples. Each step lowers the same objective, so the loop converges, but only to a local minimum that depends on the starting centroids.

## Hint 2 — formula / approach

For each iteration, compute the squared distance from every sample to every centroid, and take the argmin per sample. Then average the samples in each cluster. An empty cluster has no mean, so keep its previous centroid. Measure the largest centroid movement with a Euclidean norm, stop when it is at most `tol`, and cap the loop at `max_iter`. After the loop, assign the labels once more against the returned centroids, and compute the inertia from them.

## Hint 3 — code shape

Validate first: `X` and `init_centroids` are finite 2-D arrays with the same number of columns, `init_centroids` has between 1 and n rows, `max_iter` is an integer of at least 1, and `tol` is a finite non-negative number. Then loop `for iteration in range(1, max_iter + 1)`: assign, update, compute the shift, record `n_iter = iteration`, and break on `shift <= tol`. After the loop, call the assignment again and return the centroids, labels, inertia and `n_iter`.
