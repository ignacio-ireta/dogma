## Hint 1 — concept

Mutual information asks how far the joint distribution of the two labels is from the product of their marginals, which is what independence would give. Each pair of labels that occurs contributes a log-ratio: its observed frequency compared with the frequency that independence predicts. Only the joint counts are needed, so one pass over the rows is enough. Pick the unit before you start: this problem asks for nats.

## Hint 2 — formula / approach

With n rows, n_ab the count of each occurring pair, and n_a and n_b the marginal counts:

I(X;Y) = Σ over occurring pairs of (n_ab / n) · ln(n · n_ab / (n_a · n_b)).

Count the pairs and the two marginals with dictionaries or `collections.Counter`. Check the input before counting: both sequences non-empty, one-dimensional, and of equal length. A useful identity to test your code against is I(X;Y) = H(X) + H(Y) − H(X,Y), where each entropy is computed from counts.

## Hint 3 — code shape

Copy each input into a list first, so the caller's data is never modified. Raise `ValueError` with a named rule for each failed check. Then build `Counter(xs)`, `Counter(ys)` and `Counter(zip(xs, ys))`. Loop over the joint counter once: for each pair (a, b) and its count n_ab, add `(n_ab / n) * math.log(n * n_ab / (count_x[a] * count_y[b]))` to a running total, and return it as a float. If a hidden test fails, read the rule its message names and compare that line with it.
