# Average precision, step-wise

Implement `average_precision(y_true, scores)`. It returns the average precision (AP) of a ranking: the precision-recall curve summed in steps, with no interpolation between points. The value matches `sklearn.metrics.average_precision_score`.

## Contract

- `y_true`: 1-D labels, each 0 or 1. `scores`: 1-D finite real numbers. Both have the same length n, and at least one sample is positive.
- Let t_1 > t_2 > ... > t_m be the distinct score values, highest first. At each t_k, predict positive every sample whose score is greater than or equal to t_k. Let P_k be the precision and R_k the recall of those predictions, with R_k = TP_k / n_pos.
- AP = sum over k of (R_k - R_{k-1}) P_k, where R_0 = 0. This is a sum of rectangles of width equal to the recall gained, each at the precision reached at its threshold.
- Tied samples share one threshold. They are never split.
- Two scores are tied only when they are exactly equal as floats. No tolerance is applied, so the value is
  unchanged when every score is multiplied by the same positive constant, even 1e-9.
- Returns a Python float.
- Raises `ValueError` for unequal lengths, empty input, labels other than 0 and 1, non-finite scores, and input with no positive sample.
- Must not mutate its inputs.

## Examples

```python
>>> average_precision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
0.8333333333333333     # thresholds 0.8, 0.4, 0.35, 0.1: (0.5)(1) + (0)(0.5) + (0.5)(2/3) + (0)(0.5)
```

```python
>>> average_precision([0, 1, 0], [0.0, 0.0, 0.0])
0.3333333333333333     # one tied threshold: P = 1/3 and R = 1, so AP is the prevalence
```

```python
>>> average_precision([1, 0, 1, 0, 0], [0.9, 0.8, 0.7, 0.6, 0.5])
0.8333333333333333     # the plain mean of the five precisions would be 0.613: the recall-gain weights matter
```

On raw `breast_cancer` (569 rows, 212 malignant, so prevalence 0.373), with `malignant` as the positive
class and `worst concave points` as the score, AP is 0.957. Scaling those scores by 1e-9 must leave it at 0.957.

## Constraints

- Allowed: the Python standard library and NumPy.
- Not allowed: `sklearn`. Do not call `average_precision_score` or `precision_recall_curve`. The point is the step sum.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. They compare with scikit-learn's `average_precision_score` on tie-heavy and continuous random data, and with an explicit sum over the distinct thresholds. They check that tied scores are not split, that a constant score gives the prevalence, that a perfect ranking gives 1.0, that the explicit example gives 5/6 (and not the trapezoidal value of the same curve), that the value is a recall-weighted sum rather than a plain mean of precisions, that the result is unchanged by a strictly increasing affine change of the scores, and that missing positives, NaN, invalid labels and mismatched lengths raise `ValueError`. They also check that the caller's arrays are unchanged. Realism families covered: a raw real feature at one positive in twenty,
checked against scikit-learn, and a tiny-scale invariance that rejects any absolute tolerance merging thresholds.
