# Cross-entropy from logits

Implement `cross_entropy_from_logits(logits, targets)` for a categorical model. Each row of `logits` holds
the $k$ scores of one example, and `targets` gives the index of the true class for each row. The function returns
the mean negative log-likelihood and its gradient with respect to the logits.

With row $i$ of the logits $\mathbf{z}_i$ and true class $t_i$, let $\mathbf{p}_i = \operatorname{softmax}(\mathbf{z}_i)$. Then

$$L = -\frac{1}{n}\sum_{i=1}^{n}\log p_{i,t_i}, \qquad
\frac{\partial L}{\partial \mathbf{z}_i} = \frac{1}{n}\,(\mathbf{p}_i - \mathbf{e}_{t_i}),$$

where $\mathbf{e}_{t}$ is the one-hot vector of class $t$. The loss must be computed from the logits. Taking the log of
a softmax that has already been rounded fails for confident predictions.

## Contract

- `logits`: 2-D array-like of shape `(n, k)` with $n \ge 1$ and $k \ge 2$. Every entry is finite.
- `targets`: 1-D array of shape `(n,)` with an **integer** dtype. Every entry lies in `[0, k)`.
- Returns: a tuple `(loss, grad)`. `loss` is a Python `float`. `grad` is a `float64` array of shape `(n, k)`.
- Each row of `grad` sums to zero. Representable mean losses must remain finite even when an intermediate sum or an individual logit difference would overflow. Return `+inf` without a warning only when the mean loss itself exceeds float64 range.
- Raises `ValueError` when: `logits` is not a non-empty 2-D array; `k < 2`; any logit is NaN or infinite;
  `targets` is not 1-D of length `n`; `targets` does not have an integer dtype; or any target is outside `[0, k)`.
- Must not mutate its inputs. Must not emit a NumPy overflow, invalid-value or divide warning for any finite logits.

## Examples

```python
>>> loss, grad = cross_entropy_from_logits(np.zeros((2, 2)), np.array([0, 1]))
>>> round(loss, 6), grad
(0.693147, array([[-0.25,  0.25],
       [ 0.25, -0.25]]))
>>> loss, _ = cross_entropy_from_logits(np.array([[2.0, 1.0, 0.1]]), np.array([0]))
>>> round(loss, 6)
0.41703
>>> cross_entropy_from_logits(np.array([[0.0, 1.0]]), np.array([0.0]))
ValueError: targets must be integer class indices
```

## Constraints

- Allowed: NumPy and the standard library.
- Not allowed: `scipy`, `sklearn`, `torch`. Write the log-sum-exp and the gradient yourself.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They check: agreement with a log-sum-exp reference
on random data; the gradient against central differences of the loss; finite losses and gradients for logits of
magnitude $10^4$, with no warning, and with the correct and the wrong class at the extremes; the mean, not the
sum, so that repeating a row leaves the loss unchanged; rows of the gradient summing to zero; two-class
problems; integer logits; every rejection family (shapes, $k < 2$, empty batches, NaN and infinite logits,
float and boolean targets, out-of-range targets); and that the inputs are left unchanged.

The hidden tests draw their bulk inputs from the real datasets `breast_cancer` and `digits` (see
`docs/DATASETS.md`).
