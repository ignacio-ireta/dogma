# Entropy of a discrete distribution

Compute the Shannon entropy of a discrete probability vector, $H(\mathbf p)=-\sum_i p_i\log_b p_i$, in the unit set by the base $b$. Terms with $p_i=0$ contribute nothing, so $0\log 0$ is taken as $0$. Implement `entropy(p, base=2.0)`. The input must be a probability vector: non-negative entries that sum to 1 within $10^{-9}$. Reject anything else instead of repairing it, because a caller that passes unnormalised weights has a bug.

## Contract
- `p`: 1-D array-like of length $k \ge 1$. Every entry is finite and $\ge 0$. The entries sum to 1 within $10^{-9}$.
- `base`: finite real number with $b > 0$ and $b \ne 1$. The default `2.0` gives bits; `math.e` gives nats. For $0 < b < 1$ the formula gives a non-positive value, as the definition does.
- Returns: a Python `float`, $-\sum_i p_i \log_b p_i$.
- Raises: `ValueError` when `p` is empty, not 1-D, contains a negative or non-finite entry, or does not sum to 1 within $10^{-9}$; and when `base` is not finite, not greater than 0, or equal to 1.
- Must not mutate its inputs.

## Examples

```python
>>> entropy([0.5, 0.5])
1.0
>>> entropy([0.5, 0.25, 0.25])
1.5
>>> entropy([1.0, 0.0])
0.0
>>> round(entropy([0.25, 0.25, 0.25, 0.25], base=math.e), 4)
1.3863
```

## Constraints

- Allowed: NumPy and the standard library (`math`).
- Not allowed: `scipy` (including `scipy.stats.entropy`) and `sklearn`. Write the sum and the change of base yourself.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: agreement with a reference entropy on random distributions in three bases; zero-probability entries contributing nothing rather than `nan`; a certain outcome and a uniform distribution, both with analytic values; the relation between bases; a base below 1 giving the negated value; the sum tolerance on both sides of $10^{-9}$; rejection of negative, non-finite, empty, two-dimensional and wrong-sum input; rejection of invalid bases; tiny probabilities staying finite; a `float` return; and inputs left unchanged.

The hidden tests draw their bulk inputs from the real datasets `adult` and `breast_cancer` (see
`docs/DATASETS.md`).
