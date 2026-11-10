# Binary scores from counts

Implement `binary_scores(tp, fp, fn, tn)`. It takes the four cells of the 2 x 2 confusion matrix of one binary classifier at one threshold, and returns six scores: accuracy, precision, recall, specificity, F1 and the Matthews correlation coefficient (MCC). The function sees counts only. There are no labels and no scores.

## Contract

- `tp`, `fp`, `fn`, `tn`: non-negative integers, Python or NumPy integer types. A float is not a count even when it is whole: `2.0` raises `ValueError`, like `2.5`. Not all four may be zero.
- Returns: a `dict` with exactly the keys `"accuracy"`, `"precision"`, `"recall"`, `"specificity"`, `"f1"` and `"mcc"`, each a float.
- With N = tp + fp + fn + tn, the definitions are:
  - accuracy = (tp + tn) / N
  - precision = tp / (tp + fp)
  - recall = tp / (tp + fn)
  - specificity = tn / (tn + fp)
  - f1 = 2 tp / (2 tp + fp + fn)
  - mcc = (tp tn - fp fn) / sqrt((tp + fp)(tp + fn)(tn + fp)(tn + fn))
- Zero-division convention: a ratio whose denominator is 0 returns `0.0`. This covers precision with no predicted positives, recall with no actual positives, specificity with no actual negatives, F1 when 2 tp + fp + fn = 0, and MCC when any of its four marginal sums is 0. Never return NaN.
- Counts can reach millions, so intermediate products must not overflow a 64-bit integer.
- Raises `ValueError` for a negative count, a non-integer count, or when all four counts are 0.

## Examples

```python
>>> s = binary_scores(3, 1, 2, 4)
>>> round(s["precision"], 4), round(s["recall"], 4), round(s["f1"], 4), round(s["mcc"], 4)
(0.75, 0.6, 0.6667, 0.4082)
```

```python
>>> s = binary_scores(0, 0, 10, 990)   # fraud: 10 positives in 1000, the model never flags one
>>> s["accuracy"], s["recall"], s["specificity"]
(0.99, 0.0, 1.0)
```

```python
>>> binary_scores(0, 0, 0, 5)          # no positives anywhere: every undefined ratio is 0.0
{'accuracy': 1.0, 'precision': 0.0, 'recall': 0.0, 'specificity': 1.0, 'f1': 0.0, 'mcc': 0.0}
```

Multiplying all four counts by the same positive integer leaves each score unchanged, so counts in the
millions give the same rates as counts in the tens.

## Constraints

- Allowed: the Python standard library and NumPy.
- Not allowed: `sklearn`. Its metrics are the reference you are checked against, so do not call them.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. Those check agreement with scikit-learn on count tuples rebuilt into label vectors, with the zero-division convention applied. They check that specificity is the true negative rate and not the negative predictive value, that accuracy is not balanced accuracy, and that every zero-denominator case returns `0.0` and never NaN. They check MCC on very large counts against exact integer arithmetic, the exact key set, that swapping the two classes leaves accuracy and MCC unchanged, that F1 is the harmonic mean of precision and recall, that NumPy integers are accepted, and that invalid counts raise `ValueError`. Realism families covered: rare positives, predictive rounded ties, and large NumPy integer counts.
