## Hint 1 — concept

Every mode starts from the same counts: for each class, how many samples it catches (TP), how many it wrongly claims (FP) and how many it misses (FN). The modes differ only in how the per-class results are combined. Macro gives each class equal weight. Micro pools the counts first. Weighted weights each class by how common it is.

## Hint 2 — formula / approach

Compute `P_c = TP / (TP + FP)`, `R_c = TP / (TP + FN)` and `F1_c = 2 TP / (2 TP + FP + FN)` for each class, using 0.0 when a denominator is 0. Macro is the plain mean of each per-class value. Micro sums the counts over classes and divides once. Weighted is a mean with weights equal to each class's count in `y_true`, divided by the number of samples. Binary uses only the counts of `pos_label`.

## Hint 3 — code shape

Validate first: an unknown `average`, unequal lengths, empty input. Build the class list as `sorted(set(y_true) | set(y_pred))`. Write one helper that returns `(tp, fp, fn)` for a label and one that turns counts into `(precision, recall, f1)`. Then branch on `average`, and return a tuple of three floats.
