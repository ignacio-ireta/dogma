# Stable softmax

Implement `softmax(logits, axis=-1)`, which turns a vector of real scores (logits) into a probability
distribution along one axis: $\operatorname{softmax}(\mathbf{z})_j = e^{z_j} / \sum_k e^{z_k}$. The naive formula
overflows for large logits, and a row of huge logits must not disturb any other row. Subtract the maximum
along the axis before exponentiating, so that every exponent is at most zero.

## Contract

- `logits`: array-like of finite real numbers with at least one dimension. Any shape is valid.
- `axis`: integer, default `-1`. The normalisation runs along this axis, and every other axis is independent.
- Returns: `float64` NumPy array with the same shape as `logits`. Along `axis`, every slice is non-negative and sums to 1.
- Invariance: adding a constant to every logit in one slice leaves that slice unchanged.
- Raises `ValueError` when: `logits` is a scalar (0-d); any entry is NaN or infinite; the length along `axis`
  is zero; or `axis` is out of range.
- Must not mutate its input. Must not emit a NumPy overflow, invalid-value or divide warning for any finite input.

## Examples

```python
>>> softmax(np.array([1.0, 2.0, 3.0]))
array([0.09003057, 0.24472847, 0.66524096])
>>> softmax(np.array([1e4, 1e4 - 1.0]))           # no NaN, no warning
array([0.73105858, 0.26894142])
>>> softmax(np.array([[1.0, 2.0], [3.0, 4.0]]), axis=0).sum(axis=0)
array([1., 1.])
>>> softmax(np.array([0.0, 0.0, 0.0, 0.0]))
array([0.25, 0.25, 0.25, 0.25])
```

## Constraints

- Allowed: NumPy and the standard library.
- Not allowed: `scipy`, including `scipy.special.softmax`. The stable form is the point of this exercise.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: agreement with an independent
reference on random arrays of several shapes and along several axes; square matrices, where a wrongly
broadcast maximum would be silently wrong; extreme logits of magnitude $10^4$ with no warning or NaN;
rows with very different scales, which must not affect each other; shift invariance per slice; rows that
sum to 1; temperature scaling against the reference, with its limits; the two-class case, which must
equal the logistic function; integer input; empty and scalar inputs, NaN and infinity, and an invalid axis;
and that the input is left unchanged.

The hidden tests draw their bulk inputs from the real dataset `breast_cancer` (see
`docs/DATASETS.md`).
