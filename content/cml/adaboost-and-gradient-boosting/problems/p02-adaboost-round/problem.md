# One round of discrete AdaBoost

AdaBoost fits weak classifiers one after another. After each round it raises the weight of the samples the last classifier got wrong, so the next classifier must focus on them. Implement `adaboost_round(y, pred, weights)`, which performs the weight arithmetic of one round: it returns the classifier's vote `alpha` and the updated sample weights.

### The round
Labels are $y_i\in\{-1,+1\}$ and predictions are $\hat y_i\in\{-1,+1\}$. Start from the weights $w_i$, normalised so that $\sum_i w_i=1$. Then:

- Weighted error: $\varepsilon=\sum_i w_i\,[\hat y_i\ne y_i]$.
- If $\varepsilon\ge 0.5$, the classifier is no better than chance. Return $\alpha=0$ and the normalised weights, unchanged. The caller should stop.
- If $\varepsilon=0$, the classifier is perfect and $\alpha$ would be infinite. Raise `ValueError`, and the caller should stop.
- Otherwise: $\alpha=\tfrac12\ln\frac{1-\varepsilon}{\varepsilon}$, and the new weights are $w_i\,e^{-\alpha\,y_i\hat y_i}$, divided by their sum so that they sum to 1.

## Contract
- `y`: 1-D array-like of labels, each exactly $-1$ or $+1$, length $n\ge1$.
- `pred`: 1-D array-like of predictions, each exactly $-1$ or $+1$, same length as `y`.
- `weights`: 1-D array-like of finite, non-negative numbers with a positive sum, same length. They need not be normalised.
- Returns: `(alpha, new_weights)`: a Python `float` and a `float64` array of length $n$ that sums to 1.
- Raises: `ValueError` for empty input, mismatched lengths, labels or predictions outside $\{-1,+1\}$, negative or non-finite weights, a zero weight sum, and the perfect-classifier case $\varepsilon=0$.
- Weights may be as large as `1e308`, and the weighted error may be as small as `1e-320`. Normalise without overflowing the sum, and keep `alpha` finite for any positive error.
- Must not mutate its inputs.

## Examples

```python
>>> alpha, w = adaboost_round([1, 1, 1, 1], [1, 1, 1, -1], [1, 1, 1, 1])
>>> round(alpha, 4), w.round(4).tolist()
(0.5493, [0.1667, 0.1667, 0.1667, 0.5])
>>> alpha, w = adaboost_round([1, -1], [1, 1], [1, 1])
>>> alpha, w.tolist()
(0.0, [0.5, 0.5])
```

The first example has $\varepsilon=0.25$. The one mistaken sample is multiplied by $e^{\alpha}=\sqrt3$ and the rest by $e^{-\alpha}=1/\sqrt3$, then the result is normalised. The second example has $\varepsilon=0.5$, so the round is rejected.

## Constraints

- Allowed: NumPy and the standard library (`math`).
- Not allowed: `sklearn` (including `AdaBoostClassifier`) and `scipy`. Write the update yourself.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: agreement with a reference that computes the mistake factors explicitly, on random weights and predictions; the rejected round at and above one half, with unchanged weights; the error of zero raising `ValueError`; the classic property that after the update the same predictions have weighted error exactly one half; invariance to rescaling the weights; normalised, positive output; rejection of bad labels, predictions, weights, lengths and empty input; and inputs left unchanged.
