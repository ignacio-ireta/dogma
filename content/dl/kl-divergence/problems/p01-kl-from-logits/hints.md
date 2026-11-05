## Hint 1 — concept

Divergence compares two distributions term by term, and each term weights the log-ratio by the
probability under P. Convert the logits to log-probabilities first, so that exponentiation never
overflows. A class with zero probability is excluded from the sum, not treated as a log of zero.

## Hint 2 — formula / approach

Let $\log p_j = z_j - \mathrm{LSE}(z)$, where $\mathrm{LSE}$ is log-sum-exp and $z$ are the logits,
and do the same for q. Then $\mathrm{KL} = \sum_j p_j(\log p_j - \log q_j)$. Two cases are
special: $p_j = 0$ contributes $0$, and $q_j = 0$ with $p_j > 0$ gives $+\infty$.

## Hint 3 — code shape

Validate and cast both inputs, then subtract each row maximum before exponentiating. Build a mask
of finite log-probabilities for P, and a mask of $-\infty$ log-probabilities for Q. Use
`np.where` to select the term, and do not compute `0 * inf` anywhere: `np.where` must choose
between the infinite case, the finite product and zero. Sum along the class axis.
