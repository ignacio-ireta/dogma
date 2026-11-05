# Best split threshold for one numeric feature

A decision tree chooses, at each node, the split that most reduces impurity. For one numeric feature $x$ and class labels $y$, find the threshold $t$ with the largest **weighted impurity decrease**. Implement `best_split(x, y, *, criterion="gini")`, which returns `(threshold, impurity_decrease)`.

The impurity of a set of labels, with class proportions $p_k$, is

- `"gini"`: $G = 1 - \sum_k p_k^2$
- `"entropy"`: $H = -\sum_k p_k \log_2 p_k$ in bits, with $0\log 0 = 0$.

The candidate thresholds are the midpoints between consecutive distinct sorted values of $x$. A threshold $t$ sends the samples with $x \le t$ to the left child and the rest to the right child. With $n$ samples, $n_L$ on the left and $n_R$ on the right, its impurity decrease is

$$\Delta(t) = I(\text{parent}) - \frac{n_L}{n}\,I(\text{left}) - \frac{n_R}{n}\,I(\text{right}).$$

## Contract
- `x`: 1-D array-like of finite real numbers, length $n \ge 1$. Values may be as large as `1e308`, so compute each midpoint without overflow.
- `y`: 1-D array-like of class labels, length $n$. Equal values are the same class; any scalar labels are allowed. Labels of different types stay distinct: `1` and `"1"` are two classes, so do not let an array conversion merge them.
- `criterion`: keyword-only, `"gini"` (default) or `"entropy"`.
- Returns: `(threshold, impurity_decrease)` as a pair of Python floats. Scan the candidates in increasing threshold order and keep the incumbent unless a later candidate's decrease is larger by more than $10^{-12}$. So a tie goes to the smallest threshold. If `x` has fewer than two distinct values, return `(None, 0.0)`.
- A split is valid whenever `x` has two distinct values, even if it does not reduce impurity. Pure labels therefore return the smallest candidate with decrease `0.0`.
- Raises: `ValueError` when `x` is empty, not 1-D or not finite; when `y` is not 1-D or has a different length from `x`; or when `criterion` is neither `"gini"` nor `"entropy"`.
- Must not mutate its inputs.

## Examples

```python
>>> best_split([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1])
(2.5, 0.5)
>>> threshold, decrease = best_split([0.0, 1.0, 2.0, 3.0], [0, 1, 1, 0], criterion="entropy")
>>> threshold, round(decrease, 4)
(0.5, 0.3113)
>>> threshold, decrease = best_split([0.0, 1.0, 2.0, 3.0], [0, 1, 1, 0])
>>> threshold, round(decrease, 4)
(0.5, 0.1667)
>>> best_split([5.0, 5.0, 5.0], [0, 1, 0])
(None, 0.0)
```

The third example is a tie: thresholds $0.5$ and $2.5$ both decrease Gini by $1/6$, so the smaller one is returned.

## Constraints

- Allowed: NumPy and the standard library (`math`, `collections`).
- Not allowed: `sklearn` (including `DecisionTreeClassifier`) and `scipy`. Sort the values once and scan the candidates with running class counts. Testing every pair of rows is correct but too slow for the hidden inputs.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: agreement with a brute-force search over every midpoint, under both criteria, on random data; agreement with a depth-one scikit-learn tree on tie-free data; the tie rule on a symmetric example; invariance to the order of the rows; duplicate values never producing a threshold between equal values; constant or single-row input; pure labels; a 30,000-row input inside the time budget; rejection of an unknown criterion, empty, non-finite, two-dimensional and mismatched input; inputs left unchanged; and, on random data, a non-negative decrease and a threshold strictly between two data values.
