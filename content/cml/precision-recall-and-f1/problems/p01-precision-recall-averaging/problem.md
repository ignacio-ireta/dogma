# Precision, recall and F1 with averaging

Implement `precision_recall_f1(y_true, y_pred, *, average="binary", pos_label=1)`. It returns the precision, recall and F1 score of predictions against labels. For a binary problem it scores one positive class. For a multi-class problem it combines the per-class scores according to `average`. The averaging rule decides what the score rewards, so the same predictions can give three different numbers.

## Contract

- `y_true`, `y_pred`: 1-D sequences of labels (integers or strings, one type per call) of the same length n, with n >= 1.
- `average`: `"binary"`, `"micro"`, `"macro"` or `"weighted"`. Any other value raises `ValueError`.
- `pos_label`: the positive class. It is used only when `average="binary"`.
- Returns: a tuple `(precision, recall, f1)` of Python floats.
- For a class c, TP_c counts the samples with true label c and predicted label c. FP_c counts the samples predicted c whose true label is not c. FN_c counts the samples whose true label is c and whose prediction is not c. Then P_c = TP_c / (TP_c + FP_c), R_c = TP_c / (TP_c + FN_c) and F1_c = 2 TP_c / (2 TP_c + FP_c + FN_c). Any ratio with a zero denominator is 0.0.
- Classes are the sorted union of the labels in `y_true` and `y_pred`. A label that is only predicted is still a class.
- `"binary"`: at most two distinct labels may appear across `y_true` and `y_pred`, otherwise `ValueError`. Only `pos_label` is scored, from its TP, FP and FN. If `pos_label` never appears, all three values are 0.0.
- `"micro"`: pool TP, FP and FN over all classes, then compute the three ratios from the pooled counts.
- `"macro"`: the unweighted mean over classes of P_c, of R_c and of F1_c. F1 is the mean of the per-class F1 values. It is not computed from the macro precision and macro recall.
- `"weighted"`: the mean over classes of P_c, R_c and F1_c, weighted by the support s_c, the number of samples whose true label is c. A class with zero support gets weight 0.
- Raises `ValueError` for an unknown `average`, unequal lengths, empty input, or more than two labels with `average="binary"`.
- Must not mutate inputs.

## Examples

```python
>>> precision_recall_f1([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1])
(0.7142857142857143, 0.8333333333333334, 0.7692307692307693)
# TP = 5, FP = 2, FN = 1: precision 5/7, recall 5/6, F1 10/13.
```

```python
>>> y_true, y_pred = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 0, 1, 2, 2, 0]
>>> precision_recall_f1(y_true, y_pred, average="macro")
(0.5444444444444444, 0.5555555555555555, 0.5424242424242425)
# Per class: precision 4/5, 1/3, 1/2; recall 4/6, 1/2, 1/2; F1 8/11, 2/5, 1/2. Macro F1 is their mean.
```

```python
>>> precision_recall_f1(y_true, y_pred, average="micro")
(0.6, 0.6, 0.6)                 # micro scores equal accuracy here: 6 of 10 correct
>>> precision_recall_f1(y_true, y_pred, average="weighted")
(0.6466666666666666, 0.6, 0.6163636363636363)
```

## Constraints

- Allowed: the Python standard library.
- Not allowed: `sklearn`. Do not call `precision_recall_fscore_support`. The averaging is the point of the exercise.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. They check agreement with a reference implementation on random multi-class labels for each averaging mode, and binary scoring with string labels and a non-default positive class. They check two identities: micro scores equal accuracy, and weighted recall equals accuracy. They check that macro F1 is the mean of the per-class F1 values on a case where the other combination gives a different number. They check that predicted-only classes are averaged, that zero denominators give 0.0 and never NaN, that binary averaging rejects more than two labels, that scores do not depend on label names, and that unknown modes, empty input, mismatched lengths and mutated inputs are handled.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `digits` (see
`docs/DATASETS.md`).
