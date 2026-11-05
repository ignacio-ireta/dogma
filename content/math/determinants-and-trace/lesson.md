---
title: Determinants and trace
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

The determinant maps a square matrix $A \in \mathbb{R}^{n\times n}$ to a number $\det A$. Three
defining rules fix it uniquely:

1. **Linear in each row.** Scaling one row by $c$ scales $\det A$ by $c$. Adding two matrices that
   differ only in one row adds their determinants.
2. **Alternating.** Swapping two rows negates $\det A$.
3. **Normalised.** $\det I = 1$.

Everything else follows from these rules. The results you use most:

| Identity | Consequence |
|---|---|
| $\det A = 0 \iff A$ is singular | the zero test for invertibility |
| $\det(AB) = \det A \cdot \det B$ | determinants multiply |
| $\det A^\top = \det A$ | rows and columns play the same role |
| $\det(cA) = c^n \det A$ | scaling the whole matrix scales $n$ rows |
| Adding a multiple of one row to another | $\det$ unchanged |
| Triangular $A$ | $\det A$ is the product of the diagonal |

**Trace** is the sum of the diagonal, $\operatorname{tr} A = \sum_i a_{ii}$. It is linear, and
$\operatorname{tr}(AB) = \operatorname{tr}(BA)$. It is not multiplicative: $\operatorname{tr}(AB)
\ne \operatorname{tr}A\cdot\operatorname{tr}B$ in general.

**Eigenvalue identities.** For any square $A$ with eigenvalues $\lambda_1,\dots,\lambda_n$ counted
with multiplicity:

$$\operatorname{tr} A = \sum_i \lambda_i, \qquad \det A = \prod_i \lambda_i.$$

## Worked example

$A = \begin{pmatrix}2 & 1\\1 & 3\end{pmatrix}$.

1. $\det A = 2\cdot 3 - 1\cdot 1 = 5$. $\operatorname{tr} A = 2 + 3 = 5$.
2. Eigenvalues solve $\lambda^2 - (\operatorname{tr}A)\lambda + \det A = 0$, that is $\lambda^2 - 5\lambda + 5 = 0$.
   So $\lambda = \tfrac{5\pm\sqrt5}{2}$.
3. Check: the sum is $5 = \operatorname{tr}A$. The product is $\tfrac{25-5}{4} = 5 = \det A$.

Now the elimination route on $B = \begin{pmatrix}1&2\\3&4\end{pmatrix}$, where the answer is known to
be $1\cdot4 - 2\cdot3 = -2$.

1. Column 1: the largest entry is $3$ in row 2. Swap rows. The sign becomes $-1$.
2. Eliminate: row 2 becomes $(0, 2 - \tfrac13\cdot 4) = (0, \tfrac23)$.
3. The pivots are $3$ and $\tfrac23$. $\det B = (-1)\cdot 3\cdot\tfrac23 = -2$, which matches the direct formula.

## Why it works

**Volume.** The unit square under $A$ becomes a parallelogram. Its area is $|\det A|$, and the sign
records orientation: positive keeps the handedness, negative flips it. In $n$ dimensions the same
holds with volumes.

**Elimination.** Adding a multiple of one row to another leaves $\det$ unchanged, by rule 1 and the
fact that a repeated row gives zero. Swapping rows flips the sign. So row-reducing $A$ to upper
triangular $U$ with $s$ swaps gives $\det A = (-1)^s \prod_i u_{ii}$. This is why elimination costs
$O(n^3)$.

**Eigenvalues.** $\det(A - \lambda I)$ is a polynomial in $\lambda$ of degree $n$ with roots
$\lambda_i$, so $\det(A - \lambda I) = \prod_i(\lambda_i - \lambda)$. Setting $\lambda = 0$ gives
$\det A = \prod_i \lambda_i$. The trace identity comes from comparing the $\lambda^{n-1}$
coefficients.

## Signal vs. noise

- **Signal:** $\det = 0$ means singular. The product and transpose identities. The eigenvalue
  identities, which turn a determinant into a statement about spectra.
- **Signal:** never compute a determinant by cofactor expansion in code. It recurses over $n!$
  terms. Elimination costs $O(n^3)$, which is the same method as `solve`.
- **Noise for now:** Sarrus' rule and hand expansion past $3\times3$. Reporting a raw determinant is
  rarely the goal, so the log form is usually what you want.

## Where it breaks

- **Scale is not conditioning.** $\det(cA) = c^n\det A$. The matrix $0.1\,I_{100}$ has
  $\det = 10^{-100}$ and is perfectly conditioned, because every singular value is $0.1$. A small
  determinant does not mean near-singular. Use the condition number.
- **Overflow and underflow.** A $1000\times1000$ matrix with entries of order 1 has a determinant of
  order $10^{\pm 1300}$ (a standard Gaussian matrix has $\log|\det A| \approx \tfrac12\log n!$). That is
  far outside float64. Accumulate $\log|\det A|$ and the sign instead. NumPy's `slogdet` does this.
- **Cancellation.** For a nearly singular matrix, $\det$ is a difference of large products. Its
  relative error can be large even when the matrix itself is accurate.

## Interview angle

"What does a determinant mean, and how would you compute it for a $1000\times1000$ matrix?" Answer
shape: it is the signed volume scaling of the linear map, and zero means the map collapses a
dimension. Compute it by LU with partial pivoting in $O(n^3)$, accumulate the sign and
$\sum\log|u_{ii}|$, and never expand cofactors. Follow-ups: why a small determinant is a poor
near-singularity warning (scale), and where the log-determinant appears (the Gaussian log-likelihood
has a $\log\det\Sigma$ term). Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/matrix-inverse-and-rank` (pivots, invertibility) and
  `cml/matrix-multiplication`.
- **Unlocks:** `cml/eigenvalues-and-eigenvectors` (eigenvalues are the roots of $\det(A - \lambda I)$),
  `cml/normal-and-multivariate-gaussian` (the $\log\det\Sigma$ term), `cml/hessians-definiteness-and-critical-points`
  (leading minors test definiteness), `math/column-space-null-space-and-independence`.
- **Lineage:** Seki Takakazu and Leibniz developed determinants independently in the late 17th century
  to solve linear systems. Cramer's rule (1750) expresses each solution as a ratio of determinants.
  It is correct but far too slow for large systems.
