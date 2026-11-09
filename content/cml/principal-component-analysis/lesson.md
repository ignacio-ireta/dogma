---
title: Principal component analysis
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Stack $n\ge2$ observations of $d$ variables into $X \in \mathbb{R}^{n \times d}$, one row per observation. Subtract
each column's mean to get the centred matrix $\tilde X$. PCA looks for unit directions along which the data
spreads most. The first principal axis is

$$w_1 = \arg\max_{\lVert w \rVert = 1} \operatorname{Var}(\tilde X w),$$

the second maximises the same quantity among directions orthogonal to $w_1$, and so on. The definitions:

| Quantity | Shape | Definition |
|---|---|---|
| Principal axes $V_k$ | $d \times k$ | first $k$ right singular vectors of $\tilde X$ (the columns of $V$) |
| Explained variance $\lambda_i$ | scalar | $s_i^2 / (n-1)$, with $s_i$ the $i$-th singular value of $\tilde X$ |
| Explained-variance ratio $r_i$ | scalar | $\lambda_i / \sum_{j=1}^{d} \operatorname{var}_{n-1}(x_j)$ |
| Scores $Z_k$ | $n \times k$ | $\tilde X V_k$, the observations on the first $k$ axes |

The axes are orthonormal and sorted by variance. An axis's sign is arbitrary, because flipping it flips its scores too,
so the implementation fixes each sign with a rule.

## Worked example

Take $X = \begin{pmatrix} -3 & -1 \\ 0 & -2 \\ 0 & 3 \end{pmatrix}$, with $n = 3$ and $d = 2$.

1. **Centre.** The column means are $(-1, 0)$, so $\tilde X = \begin{pmatrix} -2 & -1 \\ 1 & -2 \\ 1 & 3 \end{pmatrix}$.
2. **Covariance.** $\tilde X^\top \tilde X = \begin{pmatrix} 6 & 3 \\ 3 & 14 \end{pmatrix}$. Dividing by $n - 1 = 2$ gives $\Sigma = \begin{pmatrix} 3 & 1.5 \\ 1.5 & 7 \end{pmatrix}$.
3. **Eigenvectors.** $\Sigma$ has eigenvalues $7.5$ and $2.5$, with unit eigenvectors $(1,3)/\sqrt{10}$ and $(3,-1)/\sqrt{10}$. Check: $7.5 + 2.5 = 10 = 3 + 7$, the total variance.
4. **Ratios.** $7.5/10 = 0.75$ and $2.5/10 = 0.25$. One axis explains 75% of the variance.
5. **Signs.** The largest-magnitude entry of each axis is $3/\sqrt{10}$ and already positive, so no flip is needed.
6. **Scores.** On axis 1, $\tilde X (1,3)^\top / \sqrt{10} = (-5, -5, 10)/\sqrt{10} \approx (-1.581, -1.581, 3.162)$. The squares sum to $15$, so the sample variance is $15/(n-1) = 7.5$, which matches $\lambda_1$.

Adding $(10, 10)$ to every row shifts only the mean; axes, variances, ratios and scores stay the same. That invariance is what centring buys.

## Why it works

For a unit vector $w$, $\operatorname{Var}(\tilde X w) = w^\top \Sigma w$, a Rayleigh quotient. Over unit vectors, its
maximum is the largest eigenvalue, reached at the matching eigenvector. The SVD $\tilde X = U S V^\top$ gives
$\tilde X^\top \tilde X = V S^2 V^\top$, so the eigenvectors of $\Sigma$ are the right singular vectors, and
$\lambda_i = s_i^2/(n-1)$. Computing the SVD of $\tilde X$ avoids forming $\tilde X^\top \tilde X$, which squares the
condition number and loses small components (Golub and Reinsch 1970, §1.1, per the Library note;
[library:Mathematics/10-Numerical-Linear-Algebra/Golub-Kahan - Numerical SVD and Pseudoinverse 1965.md]). Take
$X = [x,\; x + 10^{-7}z]$ with $n = 200$ standard normal draws, $x$ then $z$, from `numpy.random.default_rng(0)`.
A 60-digit reference gives the small eigenvalue $\lambda_2 = 5.26\times10^{-15}$. The SVD of the centred data is
off by less than $10^{-10}$ relative; the covariance route's $\lambda_2$ is off by $3\%$ to $6\%$, depending on the
BLAS (my reproduction on two platforms). Truncating axes is a chosen approximation; this error is roundoff from
squaring, and it is avoidable.

Two consequences matter in practice:

- **Reconstruction.** Keeping $k$ axes, $\lVert \tilde X - \tilde X V_k V_k^\top \rVert_F^2 = \sum_{i>k} s_i^2$. This is
  $(n-1)$ times the discarded explained variance.
- **Whitening.** For axes with $\lambda_i>0$, dividing each score by $\sqrt{\lambda_i}$ gives components with unit variance, and they are
  uncorrelated. Whitening also boosts the low-variance directions, so it boosts noise there too.

Pearson posed the best-fit question behind PCA in 1901 [library:Mathematics/07-Multivariate-Statistics/Pearson - Principal Components 1901.md].
Hotelling named principal components and turned them into a general statistical method in 1933 [library:Mathematics/07-Multivariate-Statistics/Hotelling - Principal Components 1933.md].

## Signal vs. noise

- **Signal: always centre.** Without centring, the decomposition maximises second moments about the origin; the mean can dominate the first axis.
- **Signal: standardise when units differ.** Standardising first makes PCA equal to PCA on the correlation matrix,
  so a near-constant noise variable gets the same weight as an informative one.
  Keep raw units only when the variables share a unit and their variance is the signal.
- **Noise: the sign of an axis.** Flipping it negates the axis and its scores together. Variances and the subspace
  do not change.
- **Noise: the 95% rule.** Keeping the smallest $k$ whose cumulative ratio reaches 0.95 is a convention, not a theorem.
  A scree plot or the downstream metric is a better judge.

## Where it breaks

- **Non-linear structure.** A population uniform on the unit circle has covariance $\tfrac12 I$, so no direction
  stands out. For $n\ge3$ equally spaced points, the covariance with sample divisor $n-1$ is
  $\tfrac{n}{2(n-1)}I$. Both are isotropic: PCA cannot identify the curve from these second moments.
- **Duplicate columns.** For $X = [x, x]$ the centred covariance has a zero eigenvalue, so the first axis explains
  100% of the variance.
- **Outliers.** Variance is not robust. One extreme point can rotate the first axis; refit without suspect rows to check.
- **Interpretability.** Each axis is a weighted mix of every input variable.
- **Relevance.** The top axes maximise variance, not prediction; a classifier may need a direction PCA discards.

**Units decide the axes.** Raw `wine` (178 rows, 13 columns) has column variances from about 0.02 to about 10^5. Its first axis carries 99.8% of the variance and is almost pure `proline`. Standardising the columns, a habit many people bring from other models, gives a first axis with 36.2% of the variance, led by `flavanoids`. Neither result is a bug. PCA follows the units it is given, so decide the scaling before you read the axes.

## Interview angle

The 60-second answer: PCA finds orthogonal directions of maximum variance in the centred data, computed from the SVD of
that data. Centre always, standardise when the units differ, and choose $k$ with a downstream check. Full answers are
in `interview.md`.

## Connections

- **Prerequisites:** `cml/covariance-and-correlation` and `cml/eigenvalues-and-eigenvectors`.
- **Unlocks:** `cml/singular-value-decomposition`, the factorisation PCA runs on, and then `cml/low-rank-approximation`.
