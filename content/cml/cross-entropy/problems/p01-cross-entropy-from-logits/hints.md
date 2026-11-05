## Hint 1 — concept

Cross-entropy is the negative log-likelihood of a categorical model: average, over examples, the negative log of
the probability assigned to the true class. The gradient with respect to the logits has a famously simple form:
the predicted probabilities minus the target.

## Hint 2 — formula / approach

With $\mathbf{p}_i = \operatorname{softmax}(\mathbf{z}_i)$ and true class $t_i$, the loss is
$L = -\frac1n\sum_i\big(z_{i,t_i} - \operatorname{logsumexp}(\mathbf{z}_i)\big)$. The gradient is
$\partial L/\partial\mathbf{z}_i = (\mathbf{p}_i - \mathbf{e}_{t_i})/n$. Compute the log-sum-exp with the row maximum
subtracted, and take the true-class term from the logits, not from the probability.

## Hint 3 — code shape

Validate: `logits` is a non-empty 2-D array with `k >= 2` and finite entries; `targets` is 1-D with length `n`,
has an integer dtype (check `np.issubdtype(dtype, np.integer)`), and lies in `[0, k)`. Then compute
`shifted = z - z.max(axis=1, keepdims=True)`, the row sums of `exp(shifted)`, the log-probabilities, the loss
at `rows, targets`, and adjust probabilities at the true-class positions for the gradient. Divide by `n` before summing large row losses. An overflowing true-class gap can still have a representable batch contribution: scale its original terms before subtracting.
