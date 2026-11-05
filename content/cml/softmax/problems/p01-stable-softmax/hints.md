## Hint 1 — concept

Softmax is invariant to adding the same constant to every logit in a slice. That freedom lets you choose
the constant so that every exponent is at most zero. Work out which constant, and over which elements.

## Hint 2 — formula / approach

Let $m = \max_k z_k$ taken along `axis`. Then $\operatorname{softmax}(\mathbf{z}) = e^{\mathbf{z}-m} / \sum_k e^{z_k - m}$,
and every exponent $z_k - m$ is at most $0$. Check that the same identity holds for each slice of a
matrix, and that the maximum you subtract belongs to the slice you are normalising.

## Hint 3 — code shape

Validate first: `np.asarray(logits, dtype=np.float64)`, reject a scalar, non-finite entries, an empty axis,
and an out-of-range axis with `ValueError`. Then compute `shifted = z - np.max(z, axis=axis, keepdims=True)`,
`e = np.exp(shifted)`, and return `e / np.sum(e, axis=axis, keepdims=True)`. Keep `keepdims=True` in both
reductions, or the broadcasting will change the answer silently.
