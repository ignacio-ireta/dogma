## Hint 1 — concept

A split is judged by how much it reduces impurity, and the reduction counts each child by its share of the rows. For one numeric feature there are only a few candidates that matter: the midpoints between consecutive distinct sorted values. Sorting once turns the search into a single pass, because moving the threshold past one sample just moves that sample from the right side to the left. Decide the tie rule before you code: the contract names exactly which candidate wins when two are equal.

## Hint 2 — formula / approach

Impurity of a node: Gini is 1 − Σ pₖ², entropy is −Σ pₖ log₂ pₖ with zero proportions skipped. The decrease for a threshold t is I(parent) − (nL/n)·I(left) − (nR/n)·I(right). Candidates are midpoints (a + b) / 2 where a < b are consecutive distinct sorted values. Scan candidates in increasing order and replace the incumbent only when a later decrease is larger by more than 1e-12. Return (None, 0.0) when there are fewer than two distinct values.

## Hint 3 — code shape

Validate the inputs first, with one `ValueError` per rule. Take `np.argsort` of x and reorder both x and the encoded labels. Turn the labels into class indices, build a one-hot matrix in sorted order and take its cumulative sum, so row i holds the class counts of the first i + 1 values. Find the cut positions where the sorted value strictly increases. For those cuts, the left counts are rows of the cumulative sum and the right counts are the totals minus them. Compute the impurity of each side from the counts and the sizes, combine them into the decrease, then scan the decreases in order with the tolerance rule.
