---
title: Normal equation
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Model the target as $\mathbf{y} \approx A\boldsymbol\beta$, where $A\in\mathbb{R}^{n\times p}$ is the
**design matrix** and $\boldsymbol\beta\in\mathbb{R}^p$ the coefficients. To fit an intercept, make the
first column of $A$ all ones. Least squares picks

$$\hat{\boldsymbol\beta} = \arg\min_{\boldsymbol\beta}\ \lVert\mathbf{y} - A\boldsymbol\beta\rVert^2,$$

and the **normal equations** are the system it must satisfy:

$$A^\top A\,\hat{\boldsymbol\beta} = A^\top\mathbf{y}.$$

| Object | Shape | Meaning |
|---|---|---|
| $A$ | $n\times p$ | design; column of ones first if fitting an intercept |
| $\hat{\boldsymbol\beta}$ | $p$ | coefficients (intercept first) |
| $\hat{\mathbf{y}} = A\hat{\boldsymbol\beta}$ | $n$ | fitted values: the projection of $\mathbf{y}$ onto the column space of $A$ |
| $\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}$ | $n$ | residual, orthogonal to every column of $A$ |

Three questions have separate answers. **Existence:** a least-squares fit always exists, because
$A^\top\mathbf{y}$ lies in the range of $A^\top A$, so the normal equations are always consistent.
**Uniqueness:** the coefficients are unique only with full column rank, and then
$\hat{\boldsymbol\beta} = (A^\top A)^{-1}A^\top\mathbf{y}$. **Minimum norm:** when $A$ is rank deficient,
$A^{+}\mathbf{y}$ is the least-squares solution of smallest length (per the Library note, which cites
Penrose 1955 for the generalised inverse).

## Worked example

$A = \begin{pmatrix}1&1\\1&2\\1&3\end{pmatrix}$ (intercept and one feature at $x = 1, 2, 3$) and
$\mathbf{y} = (1, 2, 2)^\top$.

1. $A^\top A = \begin{pmatrix}3&6\\6&14\end{pmatrix}$ and $A^\top\mathbf{y} = (5, 11)^\top$.
2. Solve: $3b_0 + 6b_1 = 5$ and $6b_0 + 14b_1 = 11$ give $b_1 = \tfrac12$ and $b_0 = \tfrac23$.
3. Fitted values $\hat{\mathbf{y}} = (\tfrac76, \tfrac53, \tfrac{13}{6})$, so the residual is
   $\mathbf{r} = (-\tfrac16, \tfrac13, -\tfrac16)$.
4. Check the normal equations. The ones column gives $-\tfrac16 + \tfrac13 - \tfrac16 = 0$, and the
   $x$ column gives $-\tfrac16 + \tfrac23 - \tfrac12 = 0$. Both are zero, as required.

## Why it works

Expand $L(\boldsymbol\beta) = \lVert\mathbf{y}-A\boldsymbol\beta\rVert^2 = \mathbf{y}^\top\mathbf{y} -
2\boldsymbol\beta^\top A^\top\mathbf{y} + \boldsymbol\beta^\top A^\top A\,\boldsymbol\beta$. Its gradient is
$\nabla L = 2A^\top A\boldsymbol\beta - 2A^\top\mathbf{y}$, and setting it to zero gives the normal equations.
This is the same gradient idea as `cml/gradient-descent`, solved in closed form. Because $A^\top A$ is
positive semidefinite, $L$ is convex, so the stationary point is the global minimum.

The geometry is the projection from `cml/vector-operations` (Axler 2024, §§6A–6C on minimisation, per the Library note), with the line replaced by the column space
of $A$. The residual satisfies $A^\top\mathbf{r} = 0$, so $\hat{\mathbf{y}}$ is the closest point to $\mathbf{y}$
in $\operatorname{col}(A)$. With $H = A(A^\top A)^{-1}A^\top$, the fitted values are $H\mathbf{y}$, and $H$ is
symmetric and idempotent.

**Why not invert?** $\kappa(A^\top A) = \kappa(A)^2$ in the 2-norm; Golub and Reinsch (1970, §1.1) illustrate this loss, per the Library note. Take $A$ with columns $1, t, \dots, t^6$ at 20 evenly spaced points $t \in [0,1]$, so $\kappa(A) \approx 1.8\times10^4$ and $\kappa(A^\top A) \approx 3.3\times10^8$. For noise-free $\mathbf{y} = A\mathbf{1}$, the normal-equation solution has a worst-entry error near $2\times10^{-8}$, while `np.linalg.lstsq` has one near $10^{-12}$ (my reproduction; the exact values move with the BLAS). QR factorisation (`cml/gram-schmidt-and-qr-decomposition`) and the SVD solve the same problem without forming $A^\top A$.

## Signal vs. noise

- **Signal:** the derivation (gradient equals zero), the residual being orthogonal to the columns, the rank
  condition for uniqueness, and the squared condition number.
- **Signal:** the check $A^\top\mathbf{r} = 0$ is a free test of your code.
- **Noise for now:** the formula $(A^\top A)^{-1}$ is for theory only. Use `solve` or `lstsq`. Generalised
  inverses are noise, except as the minimum-norm answer above.

## Where it breaks

- **Collinearity and the dummy-variable trap.** A one-hot encoding plus an intercept is rank deficient.
  A plain solve either raises `LinAlgError` or returns arbitrary finite coefficients. Take an intercept
  and one indicator per level, with levels $0,1,2,0,1,2$ and $\mathbf{y} = (1.25, 2.25, 3.25, 1.25, 2.25, 3.25)$.
  `solve` on the normal equations gives $(-2.75, 4, 5, 6)$ and `lstsq` gives $(1.69, -0.44, 0.56, 1.56)$
  (my reproduction). Both fit each level at the same value ($1.25$, $2.25$, $3.25$), so the coefficients
  are not identifiable, though the fitted values can stay stable. Check `np.linalg.matrix_rank(A)` against
  the number of columns first.
- **Ill-conditioned but full rank.** The normal equations lose about $2\log_{10}\kappa(A)$ digits. Take
  the 20-point design with columns $1$, $t$ and $t+10^{-8}\sin 7t$, and exact $\mathbf{y} = 1 + t$, so
  the exact fit is $(1, 1, 0)$. `solve` on the normal equations returned $(1, 0.794, 0.206)$ with
  residual $5\times10^{-9}$ on one platform and $(1, -0.035, 1.035)$ with residual $3\times10^{-8}$ on
  another, while `lstsq` returned about $(1, 1, 10^{-8})$ with residual $10^{-15}$ on both (my
  reproduction). The split between the last two coefficients is rounding noise, which is why it changes
  with the BLAS. The residual is nearly flat along the nearly parallel direction, so a residual check
  misses the error. QR avoids squaring $\kappa$, but it cannot remove the problem's own
  sensitivity (per the Library note). Use QR or `lstsq` when the columns are nearly parallel.
- **Too few rows.** When $n < p$ the system is underdetermined and the rank cannot reach $p$.
- **Cost.** Forming $A^\top A$ takes $O(np^2)$ and solving $O(p^3)$; iterative methods suit very wide designs.
- **Outliers.** Squared error gives large residuals a large pull. See `cml/regression-metrics-and-robust-losses`.

**Real data: squared conditioning.** Longley (16 rows, six macroeconomic columns in very different units) has a design with condition number about $5\times10^{9}$. The Gram matrix squares that to about $2\times10^{19}$, past $1/\varepsilon \approx 4.5\times10^{15}$. `matrix_rank` of the Gram matrix reports 6 of 7 columns, while the design itself is full rank (7 of 7). Solving the normal equations still matches the SVD fit to about $3\times10^{-11}$ in the fitted values, but the coefficients agree only to about 9 digits ($2.6\times10^{-9}$ relative). Check rank on the design, and compare fits rather than coefficients.

## Interview angle

"Derive the least-squares solution, and say when you would not use it." Answer shape: the gradient
of the squared error is zero at the normal equations. Geometrically, the fit is the projection of $\mathbf{y}$
onto the column space, and the residual is orthogonal to it. The solution is unique only with full column rank.
Then the trade-offs: the squared condition number, the cost at large $p$, and collinearity. Follow-ups: why
$X^\top X$ squares the condition number, and the dummy-variable trap. Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/vector-operations` (projection), `cml/matrix-inverse-and-rank` (rank and solving),
  `cml/partial-derivatives-and-gradients` (the gradient that gives the equations).
- **Unlocks:** `cml/gram-schmidt-and-qr-decomposition` (the stable route), `cml/orthogonal-projection`,
  `cml/ridge-regression-and-weight-decay` (the fix for rank deficiency), `cml/regression-metrics-and-robust-losses`.
- **Lineage:** Legendre published least squares in 1805, and Gauss's 1809 treatment used the normal equations.
