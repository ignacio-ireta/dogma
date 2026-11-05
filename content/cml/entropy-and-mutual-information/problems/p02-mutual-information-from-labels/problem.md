# Mutual information from two label sequences

Given two discrete label sequences $x$ and $y$ of the same length, compute the mutual information $I(X;Y)$ from their joint count table. Let $n$ be the number of rows, $n_{ab}$ the number of rows with $x=a$ and $y=b$, $n_a$ and $n_b$ the marginal counts. Then

$$I(X;Y)=\sum_{a,b:\;n_{ab}>0}\frac{n_{ab}}{n}\,\ln\frac{n\,n_{ab}}{n_a\,n_b}.$$

The result is in **nats** (natural logarithm), not bits. Labels are compared by equality, so strings and numbers both work. Implement `mutual_information(x, y)`.

## Contract
- `x`, `y`: one-dimensional sequences (a list, a tuple or a 1-D array) of scalar labels, both of length $n \ge 1$. Labels are compared by equality.
- Returns: a Python `float`, $I(X;Y)$ in nats. It is zero for independent labels and is never negative beyond floating-point rounding.
- Raises: `ValueError` when either sequence is empty, when a sequence is not one-dimensional, or when the two lengths differ.
- Must not mutate its inputs.

## Examples

```python
>>> mutual_information([0, 0, 1, 1], [0, 0, 1, 1])
0.6931471805599453
>>> round(mutual_information([0, 0, 1, 1], [0, 1, 0, 1]), 12)
0.0
>>> round(mutual_information(["a", "a", "b", "b"], ["x", "x", "x", "x"]), 12)
0.0
```

## Constraints

- Allowed: the standard library (`collections`, `math`) and NumPy.
- Not allowed: `sklearn` (including `sklearn.metrics.mutual_info_score`) and `scipy`. Count the joint table yourself.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: agreement with a reference mutual information on random string and integer labels; symmetry in the two arguments; invariance to renaming the labels; the entropy of the label for identical labels; zero for independent and for constant labels; a deterministic non-linear dependence where Pearson correlation is zero; rejection of empty, unequal-length, and two-dimensional input; a large input counted within the time budget; a property check that the value lies between zero and the smaller entropy; a `float` return; and inputs left unchanged.
