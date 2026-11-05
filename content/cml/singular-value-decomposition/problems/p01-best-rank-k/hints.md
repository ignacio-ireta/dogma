## Hint 1: concept

The SVD writes a matrix as a sum of rank-one terms, sorted from largest to smallest. Keeping the largest terms keeps
the most important structure. Because the singular vectors are orthonormal, the discarded terms are orthogonal to the
kept ones, so the error is the length of what you dropped.

## Hint 2: formula / approach

Let $A = U \Sigma V^\top$ be the economy SVD, with singular values $s_1 \ge s_2 \ge \dots$. The truncation is
$A_k = \sum_{i \le k} s_i u_i v_i^\top$. The Frobenius error is $\lVert A - A_k \rVert_F = \sqrt{\sum_{i > k} s_i^2}$.
Check the edge cases: $k = 0$ keeps nothing, and the sums over empty ranges are zero.

## Hint 3: code shape

Validate first: 2-D, at least one row and column, finite values, and an integer $k$ that is not a boolean and lies in
$[0, \min(m, n)]$. Then:

```python
u, s, vt = np.linalg.svd(a, full_matrices=False)
# keep the first k columns of u scaled by s, times the first k rows of vt
# error = square root of the sum of s[k:] squared
```
