---
title: Matrix inverse and rank
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

For a square matrix $A \in \mathbb{R}^{n\times n}$, the inverse $A^{-1}$ satisfies
$A A^{-1} = A^{-1} A = I$. It exists exactly when five statements hold together (the invertible
matrix theorem), and each gives a test you can run:

| Statement about $A$ | Test you can run |
|---|---|
| $A$ is invertible | Gauss-Jordan finds a usable pivot in every column |
| $\operatorname{rank}(A) = n$ | row reduction produces $n$ pivots |
| $\det(A) \ne 0$ | the determinant is nonzero (see `math/determinants-and-trace`) |
| $0$ is not an eigenvalue | every eigenvalue is nonzero |
| $A\mathbf{x} = \mathbf{b}$ has one solution for every $\mathbf{b}$ | the columns are independent and span $\mathbb{R}^n$ |

**Rank** is the number of pivots after row reduction. Equivalently it is the dimension of the column
space, the largest number of linearly independent columns, and the largest number of independent
rows. For an $m \times n$ matrix, $\operatorname{rank}(A) \le \min(m, n)$. "Full rank" means the
rank equals $\min(m, n)$.

**Gauss-Jordan** row-reduces the augmented matrix $[A \mid I]$ until the left block is $I$. The
right block then holds $A^{-1}$. Three row operations are allowed: swap two rows, scale a row by a
nonzero number, and add a multiple of one row to another. **Partial pivoting** means: in each
column, swap in the row (at or below the current one) with the largest absolute entry before
eliminating.

## Worked example

$A = \begin{pmatrix}2 & 1\\4 & 3\end{pmatrix}$, so $\det A = 6 - 4 = 2$.

$$[A \mid I] = \left(\begin{array}{cc|cc}2&1&1&0\\4&3&0&1\end{array}\right)$$

1. **Pivot.** The largest entry in column 1 is $4$, in row 2. Swap: $\left(\begin{array}{cc|cc}4&3&0&1\\2&1&1&0\end{array}\right)$.
2. **Eliminate below.** $R_2 \leftarrow R_2 - \tfrac12 R_1$ gives $\left(\begin{array}{cc|cc}4&3&0&1\\0&-\tfrac12&1&-\tfrac12\end{array}\right)$.
3. **Scale pivots to 1.** $R_1 \leftarrow R_1/4$ and $R_2 \leftarrow -2R_2$ give $\left(\begin{array}{cc|cc}1&\tfrac34&0&\tfrac14\\0&1&-2&1\end{array}\right)$.
4. **Eliminate above.** $R_1 \leftarrow R_1 - \tfrac34 R_2$ gives $\left(\begin{array}{cc|cc}1&0&\tfrac32&-\tfrac12\\0&1&-2&1\end{array}\right)$.

So $A^{-1} = \begin{pmatrix}\tfrac32 & -\tfrac12\\ -2 & 1\end{pmatrix}$. Check the first entry:
$2\cdot\tfrac32 + 1\cdot(-2) = 1$.

Contrast $\begin{pmatrix}1&2\\2&4\end{pmatrix}$. After eliminating, the second row becomes $(0, 0 \mid -2, 1)$.
Column 2 has no pivot, so the rank is 1 and there is no inverse.

## Why it works

Each row operation is left-multiplication by an invertible elementary matrix $E_k$. Applying the
whole sequence gives $E_k\cdots E_1 [A \mid I] = [I \mid M]$ with $MA = I$, so $M = A^{-1}$. The
process fails exactly when some column has no usable pivot. That column is then a combination of
earlier columns, which is linear dependence, which is rank deficiency. Elimination either finds $n$
pivots (invertible, full rank) or exposes a dependence. The two topics are one fact seen twice.

Partial pivoting bounds every multiplier by 1 in absolute value. Without it, a tiny pivot divides a
row, and rounding error grows with every step. It does not bound the growth of the entries: Wilkinson's
matrix reaches $2^{n-1}$ (my check).

## Signal vs. noise

- **Signal:** the equivalences. "Invertible, full rank, independent columns" is the interview core.
- **Signal:** you almost never need $A^{-1}$ itself, except for small matrices or Mahalanobis covariances. To solve $A\mathbf{x}=\mathbf{b}$, factor $A$
  (LU, via `np.linalg.solve`). For overdetermined least squares, use QR or `lstsq`. Forming the
  inverse costs about as much and is less accurate.
- **Noise for now:** the adjugate formula, block inversion and fast matrix-inversion algorithms.

## Where it breaks

- **Nearly singular is the real danger.** A matrix with a tiny nonzero pivot can be invertible in exact
  arithmetic yet dominated by input rounding. The relative error in the solution can grow by up to
  $\kappa(A) = \lVert A\rVert\,\lVert A^{-1}\rVert$ times the relative input error. The $10\times10$
  Hilbert matrix has $\kappa \approx 1.6\times10^{13}$, so $\kappa\varepsilon \approx 3.6\times10^{-3}$
  leaves two or three correct digits in float64. The threshold in `p01` (pivot $\le 10^{-12}\max|A|$)
  decides rejection, not accuracy: for $\begin{pmatrix}1&1\\1&1+10^{-10}\end{pmatrix}$ the inverse has
  entries near $10^{10}$, and one ulp in the bottom-right entry moves the $(1,1)$ entry by about
  $2.2\times10^{4}$ (my reproduction).
- **Rank needs a tolerance that scales with $A$.** Rounding rarely gives exact zeros, so `p02` treats a
  pivot as zero when it is at most $1000\,\max(m,n)\,\varepsilon\,\lVert A\rVert_F$. In 5,700 rank-deficient
  products (sizes $2$ to $10$, seed 0), a factor of 1 miscounted about 1.5% (the exact count moves
  with the BLAS: 84 on one platform, 89 on another), and the factor 1000 miscounted one on both: a
  $7\times3$ product whose elimination residue, $1.05\times10^{-11}$, exceeded its $9.0\times10^{-12}$
  tolerance (my reproduction). Cancellation can beat the margin. Nearly collinear columns show the stakes:
  $\begin{pmatrix}1&1\\1&1+10^{-13}\end{pmatrix}$ has rank 2 (NumPy agrees), but `p02` returns 1, since
  its tolerance is about $9\times10^{-13}$. The count is a decision about noise, not a fact about the
  matrix.
- **Design matrices in regression.** Collinear features make $X^\top X$ singular. One-hot columns plus
  an intercept do this (the dummy-variable trap). Drop a level, or add a ridge term $\lambda I$, which
  makes $X^\top X$ invertible for any $\lambda > 0$. Before that fix, the fitted values can stay stable
  even though the coefficients are not identifiable.
- **Rectangular and zero matrices** have no inverse; use the pseudo-inverse.

## Interview angle

"How do you solve $A\mathbf{x}=\mathbf{b}$, and do you compute the inverse?" Answer shape: no. I
factor $A$ once (LU with partial pivoting, $O(n^3)$) and solve. The factorisation is reused for
every right-hand side. An explicit inverse costs the same order, is less stable, and invites the
mistake of multiplying by it. For rectangular systems, use QR or `lstsq`. Follow-ups: when an
inverse is legitimate, and what a near-zero pivot means. Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/matrix-multiplication`, `cml/matrix-shapes-and-transpose`.
- **Unlocks:** `cml/solving-linear-systems`, `cml/lu-and-cholesky-decomposition`,
  `cml/normal-equation`, `cml/ridge-regression-and-weight-decay`, `math/column-space-null-space-and-independence`.
- **Lineage:** elimination appears in the Chinese *Nine Chapters on the Mathematical Art*. The
  condition number was introduced in Turing's 1948 analysis of rounding in matrix processes.
- **Stability vs. conditioning.** A backward-stable solver can still have large forward error on an
  ill-conditioned problem. This is the Library note's synthesis, not a statement in Goldberg's 1991 survey,
  which covers rounding error but not stability against conditioning.
