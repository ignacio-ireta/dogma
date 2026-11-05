# Best F1 threshold

Implement `best_f1_threshold(y_true, scores)`. A classifier gives each sample a score, and you must choose the threshold above which samples are flagged as positive. This function returns the threshold that maximises F1 on the labelled data, together with the F1 it reaches. It is the operating-point search behind most "pick a cut-off" decisions. It is also where tied scores need an explicit rule.

## Contract

- `y_true`: 1-D labels, each 0 or 1 (integers or booleans). `scores`: 1-D finite real numbers. Both have the same length n, with n >= 1.
- A sample is predicted positive at threshold t when its score is greater than or equal to t.
- The candidate thresholds are the distinct score values. Tied samples always share one candidate, so a threshold never splits a tie.
- F1 at a candidate is 2 TP / (2 TP + FP + FN), computed as one division of the integer counts at that threshold.
- Returns `(threshold, f1)`, where the threshold is the candidate with the largest F1. If several candidates reach that largest F1, return the largest threshold.
- If there are no positive samples, every F1 is 0.0, so the result is `(highest score, 0.0)`.
- Raises `ValueError` for empty input, unequal lengths, input that is not 1-D, labels other than 0 and 1, and NaN or infinite scores.
- Must not mutate its inputs.
- Must run in O(n log n): sort once, then sweep the thresholds.

## Examples

```python
>>> best_f1_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
(0.35, 0.8)
# Flagging from 0.35 up: TP = 2, FP = 1, FN = 0, so F1 = 4/5.
```

```python
>>> best_f1_threshold([0, 0, 1, 1], [0.0, 0.0, 0.0, 1.0])
(1.0, 0.6666666666666666)
# Thresholds 0.0 and 1.0 both reach F1 = 2/3. The largest threshold wins.
```

```python
>>> best_f1_threshold([1, 0], [0.5, 0.5])
(0.5, 0.6666666666666666)
# Tied scores move together. Flagging both gives TP = 1, FP = 1, so F1 = 2/3.
```

## Constraints

- Allowed: the Python standard library and NumPy.
- Not allowed: `sklearn`. Do not call `precision_recall_curve` or `f1_score`. The sweep is the exercise.

## Done looks like

All public tests pass, then **Submit** passes the hidden tests. They compare the chosen threshold and F1 with a brute-force search over every candidate, on random data with and without ties. They check the returned F1 against scikit-learn's F1 at that threshold, and that no candidate beats it. They check that tied scores are never split, that the largest threshold wins a tie, that a score equal to the threshold counts as positive, that the lowest threshold (flag everything) is a candidate, and that data with no positives returns `(highest score, 0.0)`. They also check that NaN, infinity, labels outside 0 and 1, and malformed shapes raise `ValueError`, that a 200,000-score input finishes quickly, and that the caller's arrays are left unchanged.
