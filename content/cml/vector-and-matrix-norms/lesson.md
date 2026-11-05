---
title: Vector and matrix norms
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A norm $\lVert\cdot\rVert$ measures the size of a vector $\mathbf{x} \in \mathbb{R}^n$. It must satisfy three rules:

1. **Positivity:** $\lVert\mathbf{x}\rVert \geq 0$, and it is $0$ only when $\mathbf{x} = \mathbf{0}$.
2. **Homogeneity:** $\lVert\alpha\mathbf{x}\rVert = |\alpha|\,\lVert\mathbf{x}\rVert$ for every scalar $\alpha$.
3. **Triangle inequality:** $\lVert\mathbf{x} + \mathbf{y}\rVert \leq \lVert\mathbf{x}\rVert + \lVert\mathbf{y}\rVert$.

The family used in ML is the $p$-norm, for real $p \geq 1$, plus the limit $p = \infty$:

$$\lVert\mathbf{x}\rVert_p = \Big(\sum_{i=1}^{n} |x_i|^p\Big)^{1/p}, \qquad \lVert\mathbf{x}\rVert_\infty = \max_i |x_i|.$$

| Norm | Formula | Unit ball in $\mathbb{R}^2$ | Penalty behaviour |
|---|---|---|---|
| $\lVert\mathbf{x}\rVert_1$ | $\sum_i \lvert x_i\rvert$ | diamond, corners on the axes | pushes coefficients to exactly 0 |
| $\lVert\mathbf{x}\rVert_2$ | $\sqrt{\sum_i x_i^2}$ | circle | shrinks all coefficients smoothly |
| $\lVert\mathbf{x}\rVert_\infty$ | $\max_i \lvert x_i\rvert$ | square, corners on the diagonals | caps the largest entry |

For $p < 1$ the formula still runs, but it is not a norm. See *Where it breaks*.

## Worked example

Let $\mathbf{x} = (3, -4)$.

- $\lVert\mathbf{x}\rVert_1 = 3 + 4 = 7$
- $\lVert\mathbf{x}\rVert_2 = \sqrt{9 + 16} = 5$
- $\lVert\mathbf{x}\rVert_\infty = \max(3, 4) = 4$

These are ordered: $4 \le 5 \le 7$. In general $\lVert\mathbf{x}\rVert_\infty \le \lVert\mathbf{x}\rVert_2 \le \lVert\mathbf{x}\rVert_1 \le \sqrt{n}\,\lVert\mathbf{x}\rVert_2$, and a larger $p$ never gives a larger norm.

**Triangle inequality failing at $p = \tfrac12$.** Take $\mathbf{e}_1 = (1, 0)$ and $\mathbf{e}_2 = (0, 1)$. Each has norm 1. Their sum $(1, 1)$ has $\lVert\cdot\rVert_{1/2} = (1 + 1)^2 = 4 > 2$.

**Overflow.** Take $\mathbf{x} = (3\cdot10^{200},\ 4\cdot10^{200})$. Squaring the entries gives $9\cdot10^{400}$ and $1.6\cdot10^{401}$, both above the float64 maximum of about $1.8 \times 10^{308}$, so the naive formula returns `inf`. Divide by the largest entry $s = 4\cdot10^{200}$ first. The ratios are $(0.75, 1)$, the sum of squares is $1.5625$, its root is $1.25$, and $s \cdot 1.25 = 5\cdot10^{200}$. The scaling is exact in real arithmetic. In float64 the last step prints as `4.9999999999999995e+200`, because $5\cdot10^{200}$ has no exact binary form.

## Why it works

Scaling works because of homogeneity. For any $s > 0$, $\lVert\mathbf{x}\rVert_p = s\,\lVert\mathbf{x}/s\rVert_p$. Choose $s = \max_i |x_i|$. Every ratio $|x_i|/s$ lies in $[0, 1]$, so no power can overflow, and the largest term is exactly 1, so the sum is at least 1 and cannot underflow to zero.

Why the balls differ comes down to smoothness. The L2 penalty $\lVert\mathbf{w}\rVert_2^2$ has gradient $2\mathbf{w}$, smooth everywhere. The L1 penalty has corners where a coordinate is exactly zero. When you minimise a loss plus $\lambda\lVert\mathbf{w}\rVert$, the solution is where the loss contour first touches the penalty ball. Round contours touch a corner rarely, but a diamond touches it often, and at a corner some coefficients are exactly zero. That is the sparsity of the lasso. Hastie, Tibshirani and Friedman's §3.4 shows the same picture for the family of $q$-norm penalties.

Matrix norms come from the same ideas. The **Frobenius norm** is the L2 norm of the entries, $\lVert A\rVert_F = \sqrt{\sum_{ij} A_{ij}^2} = \sqrt{\operatorname{tr}(A^\top A)}$. The **spectral norm** is the largest stretch the matrix applies to any unit vector, $\lVert A\rVert_2 = \max_{\lVert\mathbf{x}\rVert_2 = 1} \lVert A\mathbf{x}\rVert_2 = \sigma_{\max}(A)$. The two are related by $\lVert A\rVert_2 \le \lVert A\rVert_F \le \sqrt{\operatorname{rank} A}\,\lVert A\rVert_2$.

## Signal vs. noise

- **Signal:** the three norms and their unit balls. L1 gives sparsity, L2 gives smooth shrinkage.
- **Signal:** norms depend on units. A penalty on coefficients is fair only when the features are on comparable scales, so standardise first (`cml/feature-scaling`).
- **Signal:** compute overflow-safely. Divide by the largest entry before raising to a power.
- **Noise for now:** dual norms, Schatten norms beyond the spectral and Frobenius cases, and the nuclear norm (the sum of singular values), which you will meet as a convex relaxation of rank.

## Where it breaks

- **$p < 1$ is not a norm.** The unit ball is not convex, and the triangle inequality fails (the example above). Penalties with $p < 1$ are non-convex, so gradient methods can get stuck.
- **Non-differentiable points.** $|x_i|$ has no derivative at zero, so the L1 penalty needs subgradients. The L∞ norm has ties when several entries share the maximum. The L2 norm, unsquared, has no gradient at the zero vector.
- **Overflow and underflow.** Squaring overflows once an entry passes about $1.34 \times 10^{154}$, and a tiny entry (below about $10^{-162}$) squared underflows to zero. Scale first.
- **NumPy does not scale for you.** In the NumPy this repo pins, `np.linalg.norm([3e200, 4e200])` returns `inf`, and `np.linalg.norm([3e-200, 4e-200])` returns `0.0`. Do not treat it as a reference for extreme magnitudes.
- **Frobenius is not spectral.** A matrix with many small entries can have a large Frobenius norm and a small spectral norm.

## Interview angle

*60-second shape:* ridge penalises $\lVert\mathbf{w}\rVert_2^2$ and shrinks smoothly. The lasso penalises $\lVert\mathbf{w}\rVert_1$, whose diamond-shaped ball has corners on the axes, so some coefficients land exactly on zero.

*Likely follow-ups:* correlated features (the lasso picks one of a group, and elastic net keeps groups together), and norms in deep learning (global-norm gradient clipping and spectral normalisation). Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/vector-operations` (the L2 norm as $\sqrt{\mathbf{x}\cdot\mathbf{x}}$) and `cml/matrix-multiplication` (the action $A\mathbf{x}$ behind the spectral norm).
- **Unlocks:** `cml/ridge-regression-and-weight-decay`, `cml/lasso-and-elastic-net`, `cml/eigenvalues-and-eigenvectors` (the spectral norm from $A^\top A$), and `cml/cosine-similarity` (normalising by the L2 norm).
- **Lineage:** the triangle inequality for $p \geq 1$ is Minkowski's inequality. The lasso (Tibshirani, 1996) brought the L1 penalty into regression.
