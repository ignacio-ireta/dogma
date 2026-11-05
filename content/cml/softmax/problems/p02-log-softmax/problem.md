# Log-softmax

Implement `log_softmax(logits, axis=-1)`, the logarithm of the softmax, computed directly:
$\log\operatorname{softmax}(\mathbf{z})_j = z_j - \log\sum_k e^{z_k}$. Cross-entropy losses need these
log-probabilities. Taking the log of a probability that has already underflowed to zero gives $-\infty$, and the
right answer is finite: a logit that is $10^4$ below the others has log-probability about $-10^4$. Compute the
log-sum-exp with the maximum subtracted, so that no exponent is positive and no probability is rounded first.

## Contract

- `logits`: array-like of finite real numbers with at least one dimension. Any shape is valid.
- `axis`: integer, default `-1`. Every other axis is an independent distribution.
- Returns: `float64` NumPy array with the same shape as `logits`. Along `axis`, `exp` of the result sums to 1, and every entry is at most 0.
- Accuracy: representable log-probabilities are finite, including for logits of magnitude $10^4$ and for huge common offsets. If a negative log-probability is outside float64 range, return `-inf` without a warning; this is a representability limit, not probability underflow.
- Invariance: adding a constant to every logit in one slice leaves that slice unchanged.
- Raises `ValueError` when: `logits` is a scalar (0-d); any entry is NaN or infinite; the length along `axis` is zero; or `axis` is out of range.
- Must not mutate its input. Must not emit a NumPy overflow, invalid-value or divide warning for any finite input.

## Examples

```python
>>> log_softmax(np.array([0.0, 0.0]))
array([-0.69314718, -0.69314718])
>>> log_softmax(np.array([0.0, -800.0]))            # finite; the second entry is -800
array([  0., -800.])
>>> log_softmax(np.array([2e4, 0.0]))              # the tail stays finite
array([     0., -20000.])
>>> log_softmax(np.array([1.0, 2.0]))
array([-1.31326169, -0.31326169])
>>> np.isclose(np.exp(log_softmax(np.array([1.0, 2.0, 3.0]))).sum(), 1.0)
True
```

## Constraints

- Allowed: NumPy and the standard library.
- Not allowed: `scipy`, including `scipy.special.log_softmax` and `scipy.special.logsumexp`. Writing the log-sum-exp is the exercise.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: agreement with an independent
reference on random arrays of several shapes and axes; finite values for logits of magnitude $10^4$ with no
warning, including a tail probability that a plain log of a softmax would lose; that `exp` of the result sums
to 1; that a row of huge logits does not change its neighbours; shift invariance per slice; square matrices
normalised along each axis; integer input; empty, scalar, NaN, infinite and out-of-range-axis inputs; and that
the input is left unchanged.
