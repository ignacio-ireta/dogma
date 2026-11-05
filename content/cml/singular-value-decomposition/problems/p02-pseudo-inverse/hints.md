## Hint 1: concept

Inverting a matrix means inverting its SVD, one singular value at a time. A zero singular value cannot be inverted, so it
stays zero. A tiny one can be numerically zero, so the cutoff decides which singular values are trusted. The cutoff
must scale with the largest singular value, so that a matrix's overall size does not change its answer.

## Hint 2: formula / approach

With $A = U \Sigma V^\top$, the pseudo-inverse is $A^+ = V \Sigma^+ U^\top$. Each entry of $\Sigma^+$ is $1/s_i$ when
$s_i > \texttt{rcond} \cdot s_{\max}$, and $0$ otherwise. The result has shape $(n, m)$, which is the transpose of
$A$'s shape.

## Hint 3: code shape

Validate first: 2-D, non-empty, finite values, and an `rcond` that is a finite real number $\ge 0$ and not a boolean.
Then:

```python
u, s, vt = np.linalg.svd(a, full_matrices=False)
# keep = s > rcond * s.max()
# s_plus: zeros, with 1 / s[keep] in the kept positions
# combine vt.T, scaled by s_plus, with u.T
```
