# Confusion matrix from labels

Implement `confusion_matrix(y_true, y_pred, labels=None)`. It counts how often each pair (true label, predicted label) occurs and returns the counts as a square integer matrix. Rows index the true label and columns index the predicted label, so entry `[i, j]` counts the samples whose true label is the i-th label and whose predicted label is the j-th. Every count-based score in this topic starts from this matrix. Transposing it exchanges false positives and false negatives, so precision and recall swap even though accuracy remains unchanged.

## Contract

- `y_true`, `y_pred`: 1-D sequences (lists, tuples or NumPy arrays) of labels. Labels are integers or strings, one type per call. Both have the same length n, with n >= 1.
- `labels`: `None`, or a sequence of distinct labels that sets the row and column order.
- Returns: a NumPy integer array of shape (k, k), where k is the number of labels.
- With `labels=None`, the labels are the sorted union of the labels in `y_true` and `y_pred`. Strings sort in Python's default order.
- With `labels` given, rows and columns follow that order. A sample whose true or predicted label is not in `labels` is ignored. A listed label that never occurs gets a zero row and a zero column.
- Raises `ValueError` when `y_true` and `y_pred` differ in length, when `y_true` is empty, or when `labels` contains a duplicate.
- Must not mutate inputs.

## Examples

```python
>>> confusion_matrix([0, 1, 2, 2], [0, 2, 2, 1])
array([[1, 0, 0],
       [0, 0, 1],
       [0, 1, 1]])
# Row 2 holds two true-2 samples: one predicted 2 (column 2), one predicted 1 (column 1).
```

```python
>>> confusion_matrix(["cat", "dog", "cat"], ["dog", "dog", "cat"])
array([[1, 1],
       [0, 1]])
# Labels sort to cat, dog. Row cat: one sample predicted dog, one predicted cat.
```

```python
>>> confusion_matrix([0, 1, 2, 2], [0, 2, 2, 1], labels=[2, 0, 5])
array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 0]])
# Label 1 is not listed, so the two samples that touch it are ignored. Label 5 never occurs: zero row and column.
```

## Constraints

- Allowed: the Python standard library and NumPy (for the returned array).
- Not allowed: `sklearn`. Do not call `sklearn.metrics.confusion_matrix`. The point is the counting.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with a reference implementation on random integer labels, with and without an explicit `labels` list that contains unseen labels and omits seen ones. They also cover the orientation on asymmetric data (rows are true labels), a label that is only predicted, the order given by `labels`, repeated pairs adding up, samples outside `labels` being ignored, integer dtype, string labels, duplicate and empty and mismatched inputs raising `ValueError`, and inputs left unchanged.
