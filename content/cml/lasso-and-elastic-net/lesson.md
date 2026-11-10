---
title: Lasso and elastic net
minutes: 12
min_viable: "Read The system and Worked example, then solve p01."
---

## The system

For $n$ samples and $d$ features, let $X\in\mathbb R^{n\times d}$, $\mathbf y\in\mathbb R^n$, and scalar penalty $\alpha\ge0$. The lasso minimises over $\boldsymbol\beta\in\mathbb{R}^d$ with no intercept (centre the data first),

$$\min_{\boldsymbol\beta}\ \frac{1}{2n}\lVert\mathbf y - X\boldsymbol\beta\rVert^2 + \alpha\lVert\boldsymbol\beta\rVert_1.$$

Cyclic coordinate descent updates one coefficient at a time. Let $\rho_j = \frac1n X_j^\top(\mathbf y - \sum_{k\ne j} X_k\beta_k)$
be the scalar correlation of column $j$, $X_j\in\mathbb R^n$, with what the other coordinates leave unexplained, and $z_j = \frac1n\lVert X_j\rVert^2$. The update is

$$\beta_j \leftarrow \frac{S(\rho_j,\alpha)}{z_j},\qquad S(x,\lambda) = \operatorname{sign}(x)\max(\lvert x\rvert-\lambda,0).$$

The soft-threshold $S$ sets every $\rho_j$ with $\lvert\rho_j\rvert\le\alpha$ to exactly zero. Three facts follow:

- **Exact zeros.** A coefficient is zero when its correlation is no larger than $\alpha$. L2 does not enforce sparsity; an exact zero can still occur through a zero correlation or symmetry.
- **Threshold.** With $\alpha_\max = \max_j\lvert X_j^\top\mathbf y\rvert/n$, every coefficient is zero for $\alpha\ge\alpha_\max$.
- **Optimality (KKT).** At the solution, $\frac1n X_j^\top(\mathbf y - X\boldsymbol\beta) = \alpha\operatorname{sign}(\beta_j)$ when $\beta_j\neq 0$,
  and its size is at most $\alpha$ when $\beta_j = 0$.

The elastic net adds an L2 term, $\alpha\rho\lVert\boldsymbol\beta\rVert_1 + \tfrac12\alpha(1-\rho)\lVert\boldsymbol\beta\rVert^2$, with mixing ratio
$\rho\in[0,1]$. Its update is $S(\rho_j,\alpha\rho)/\bigl(z_j+\alpha(1-\rho)\bigr)$.

## Worked example

Take $X = (1,2,3)^\top$ and $\mathbf y = (2,4,6)^\top$. Then $\rho_1 = \frac13(2+8+18) = \frac{28}{3}$ and $z_1 = \frac{14}{3}$.

- Lasso, $\alpha = 1$: $\beta = (\tfrac{28}{3}-1)/\tfrac{14}{3} = \tfrac{25}{14}\approx 1.786$.
- Lasso, $\alpha = 10$: $10 > \tfrac{28}{3}$, so $\beta = 0$ exactly. Here $\alpha_\max = \tfrac{28}{3}$.
- Elastic net, $\alpha = 1$, $\rho = 0.5$: $\beta = (\tfrac{28}{3}-0.5)/(\tfrac{14}{3}+0.5) = \tfrac{53}{31}\approx 1.710$.

Each value matches scikit-learn's solver at a tight tolerance.

**Duplicate columns.** Duplicate the example column, so $X=[x,x]$ with $x=(1,2,3)^\top$. At $\alpha=1$, every nonnegative lasso pair whose sum is $25/14$ is optimal. Cyclic descent from zero selects $(25/14,0)$ in column order. The elastic net with $\rho=0.5$ has the unique symmetric solution $(53/59,53/59)$.

## Why it works

**Why the threshold is soft.** For one coordinate, minimising $\tfrac12 z\beta^2 - \rho\beta + \alpha\lvert\beta\rvert$ gives zero while
$\lvert\rho\rvert\le\alpha$, and otherwise moves $\rho$ toward zero by $\alpha$. Dividing by $z$ undoes the column scale. The
$1/n$ in $\rho$ and $z$ puts $\alpha$ on the same scale as the mean squared error.

**Proximal view.** $S(\cdot,\lambda)$ is the proximal operator of $\lambda\lvert\cdot\rvert$, the unique minimiser of $\lambda\lvert b\rvert+\tfrac12(b-v)^2$ over $b$. Per the Library note, the definition is Parikh & Boyd §1.1 and the lasso case is §7.1 [source: library-refresh/01-Mathematics/08-Optimization/2026 -- Library -- Proximal Steps and Accelerated Convex Optimization -- Concept Note.md]. The kink is handled exactly, not smoothed.

**Shrinkage shape.** Ridge (mixing ratio $0$, same $\alpha$) multiplies a coefficient by a factor; the lasso subtracts a constant. With $z=14/3$ and $\rho=28/3$ from the example, $\rho/z = 2$ is the unpenalised slope, and $\alpha=1$. Ridge gives $2\cdot z/(z+\alpha) = 1.647$; the lasso gives $2-\alpha/z = 1.786$. For an unpenalised slope of $0.1$ the lasso returns exactly $0$, while ridge returns $0.0824$. The constant subtraction is why small coefficients vanish.

**Why the path is monotone.** For $\alpha_1<\alpha_2$, optimality of each solution against the other gives $(\alpha_2-\alpha_1)(\lVert\boldsymbol\beta^{(1)}\rVert_1 - \lVert\boldsymbol\beta^{(2)}\rVert_1)\ge 0$, so the $\ell_1$ norm never grows with $\alpha$.

**Why correlated columns fail.** If two columns are identical, the objective depends only on their sum, and every nonnegative split of that sum is optimal, so the solver's choice is arbitrary. A positive L2 part, $\alpha(1-\rho)>0$, makes the minimiser unique and gives identical columns equal weight (Zou & Hastie 2005).

## Signal vs. noise

- **Signal:** the soft-threshold, exact zeros, the threshold $\alpha_\max$, and the correlated-feature failure.
- **Signal:** the elastic net splits weight across correlated features, and it reduces to the lasso at $\rho=1$ and to ridge at $\rho=0$.
- **Noise for now:** the sweep order and tolerance, and the path algorithms (least-angle regression) that trace all $\alpha$ at once.

## Where it breaks

- **Correlated features.** Near-duplicates swap places under tiny perturbations, so the selected feature is unstable.
  For prediction the arbitrary choice often costs little; for interpretation it misleads. Elastic net, or grouping correlated columns, is the fix.
- **Gradient steps on the L1 term.** Subgradient descent generally does not return exact zeros. For $\tfrac12(b-0.3)^2+0.6\lvert b\rvert$, whose minimiser is $0$, steps of size $0.1$ from $b_0=1$ still change sign about every other step after 1,000 iterations, and none is exactly zero. The proximal step (same note, §4.2) reaches exactly $0$ at iteration 14. Use a proximal or coordinate step for the nonsmooth term.
- **Units.** The penalty depends on each column's scale. Standardise first, as for ridge.
- **Intercept.** The solver here fits none. Centre $X$ and $\mathbf y$, or the fit is forced through the origin.
- **Biased estimates.** Selected coefficients are shrunk toward zero, and standard errors after selection are not valid as usual.
- **Slow convergence.** Cyclic updates can crawl along nearly collinear columns. Check convergence at a tighter tolerance or a larger sweep budget before trusting the selected set.

**Real data: a correlated pair.** On raw diabetes (442 rows, $s_1$ and $s_2$ correlated at 0.9, column scales from 0.5 to 35), at $\alpha$ set to 15% of the largest penalty, the lasso keeps $s_1 = 0.19$ and sets $s_2 = 0$. The elastic net with $\rho = 0.5$ keeps both, at $0.47$ and $-0.31$. Which member of the pair the lasso keeps is arbitrary, and it changes with small changes in the data. Centre the columns before comparing these coefficients: the fit has no intercept.

## Interview angle

*"Why does the lasso give exact zeros and ridge does not?"* The L1 penalty has corners on the axes, where the optimum tends to
land. The soft-threshold makes the corner explicit: small correlations are set to zero. With correlated features the choice of
which one survives is unstable, so I would reach for the elastic net. Follow-ups: the lasso path and why $\alpha_\max$ is the
starting point; how to pick $\alpha$ (cross-validation on standardised features).

Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/ridge-regression-and-weight-decay` (the L2 penalty and centring), `cml/gradient-descent`
  , `cml/partial-derivatives-and-gradients`.
- **Unlocks:** `cml/hyperparameter-search` (tuning $\alpha$ and $\rho$), `cml/feature-scaling`, and `cml/aic-bic-and-stepwise-selection`
 .
- **Lineage:** Tibshirani 1996 introduced the lasso; Zou & Hastie 2005 introduced the elastic net; Friedman et al. 2007 developed the
  pathwise coordinate descent used here.
