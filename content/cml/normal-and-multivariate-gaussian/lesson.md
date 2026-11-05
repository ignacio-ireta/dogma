---
title: Normal and multivariate Gaussian
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

The one-dimensional normal with mean $\mu$ and variance $\sigma^2$ has density

$$\log p(x) = -\tfrac12\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}.$$

The $d$-dimensional version replaces the scalars with a mean vector $\boldsymbol\mu \in \mathbb{R}^d$ and a
covariance matrix $\Sigma \in \mathbb{R}^{d\times d}$, which must be symmetric and positive definite:

$$\log \mathcal{N}(\mathbf{x}\mid\boldsymbol\mu,\Sigma) = \underbrace{-\tfrac{d}{2}\log 2\pi}_{\text{constant}}
\;\underbrace{-\,\tfrac12\log\det\Sigma}_{\text{volume}}
\;\underbrace{-\,\tfrac12\,\delta^2}_{\text{shape}},
\qquad \delta^2 = (\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu).$$

Three terms, each with a job:

- **Constant:** depends only on $d$.
- **Volume term:** $\det\Sigma$ is the squared volume scale of the covariance ellipsoid. A wider distribution has
  a larger determinant and a lower peak density.
- **Mahalanobis distance** $\delta = \sqrt{\delta^2}$: distance from the mean measured in units of the
  distribution's own spread, along each principal direction. Points with equal $\delta$ lie on one ellipsoid.

Two closure facts make the Gaussian the default:

- **Marginals.** Any subset of components is Gaussian, with the matching sub-vector of $\boldsymbol\mu$ and
  sub-matrix of $\Sigma$.
- **Conditionals.** Split $\mathbf{x} = (\mathbf{x}_1,\mathbf{x}_2)$. Then $\mathbf{x}_1 \mid \mathbf{x}_2$ is Gaussian with
  mean $\boldsymbol\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(\mathbf{x}_2-\boldsymbol\mu_2)$ and covariance
  $\Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$.

## Worked example

$\boldsymbol\mu = (0,0)$, $\Sigma = \begin{pmatrix}2&1\\1&2\end{pmatrix}$, $\mathbf{x} = (1,0)$.

1. **Determinant.** $\det\Sigma = 4 - 1 = 3$, so $\log\det\Sigma = \log 3 = 1.0986$.
2. **Mahalanobis.** $\Sigma^{-1} = \tfrac13\begin{pmatrix}2&-1\\-1&2\end{pmatrix}$, so
   $\delta^2 = \tfrac13\cdot 2 = 0.6667$.
3. **Assemble.** $\log p = -\log 2\pi - \tfrac12(1.0986) - \tfrac12(0.6667) = -1.8379 - 0.5493 - 0.3333 = -2.7205$.

Now the same number without the inverse. Factor $\Sigma = LL^\top$ with $L = \begin{pmatrix}\sqrt2&0\\ 1/\sqrt2&\sqrt{3/2}\end{pmatrix}$.
Then $2\sum_i\log L_{ii} = 2(0.3466+0.2027) = 1.0986 = \log\det\Sigma$. Solving $L\mathbf{z} = \mathbf{x}$ gives
$\mathbf{z} = (0.7071, -0.4082)$, and $\lVert\mathbf{z}\rVert^2 = 0.6667 = \delta^2$.

## Why it works

The density is $(2\pi)^{-d/2}\det(\Sigma)^{-1/2}\exp(-\tfrac12\delta^2)$. Taking logs turns the product of
factors into a sum, and it protects against underflow. A 60-dimensional covariance with 60 eigenvalues evenly
spaced from $10^{-7}$ to $10^{-6}$ has $\det\Sigma \approx 10^{-380}$, below the float64 range (smallest normal
$\approx 2.2\times10^{-308}$), so `np.linalg.det` returns $0$. Its natural logarithm is $-874.0$, which is perfectly
representable.

Cholesky is the right factorisation for three reasons. It exists exactly when $\Sigma$ is positive definite,
so it doubles as the check. $\log\det\Sigma$ is twice the sum of the logs of $L$'s diagonal. And a solve with
$L$ gives $\delta^2 = \lVert\mathbf{z}\rVert^2$ without ever forming $\Sigma^{-1}$, which is both cheaper and
less rounding-prone (see `cml/matrix-inverse-and-rank`). This is standard numerical practice (Trefethen &
Bau, *Numerical Linear Algebra*, lectures on Cholesky factorization).

## Signal vs. noise

- **Signal:** the three terms, and the Mahalanobis distance as "distance in standard deviations".
- **Signal:** compute in log space with a factorisation. Never form $\det\Sigma$ or $\Sigma^{-1}$ for a density.
- **Signal:** the marginal and conditional formulas. They underlie Gaussian processes, Kalman filters and
  factor models.
- **Noise for now:** the normalising constant of other distributions, and the characteristic function.

## Where it breaks

- **Singular covariance.** If one direction has zero variance, $\det\Sigma = 0$ and the density does not exist
  on $\mathbb{R}^d$. Cholesky fails, and the function must raise. Collinear features make this happen.
- **Too few samples.** With $n < d$, the sample covariance has rank at most $n-1$. Its inverse does not exist
  unless you regularise it (shrink toward a diagonal, or add $\lambda I$).
- **Heavy tails.** The quadratic term grows like $\delta^2$, so one outlier with a large Mahalanobis distance
  dominates the log-likelihood. Check the tails before trusting a Gaussian fit.
- **Symmetry is not optional.** Cholesky reads one triangle. An asymmetric matrix is silently replaced by a
  different one unless you check first.

## Interview angle

"Write the log-density of a multivariate normal and compute it stably." Answer shape: the three terms, then
the implementation. Factor $\Sigma = LL^\top$, sum the logs of $L$'s diagonal for the log-determinant, solve
for the quadratic form, and never invert. Follow-ups: "What is the Mahalanobis distance and when does it beat
Euclidean distance?" (It accounts for correlation and scale. Euclidean treats all directions alike.) "Given a
joint Gaussian, what is the distribution of one block given the other?" (The conditional formula above.) Full
answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/covariance-and-correlation` (the matrix $\Sigma$), `cml/matrix-inverse-and-rank`
  (solving with a factorisation).
- **Unlocks:** `cml/maximum-likelihood-and-map` (the Gaussian log-likelihood is the object being maximised),
  and `cml/lu-and-cholesky-decomposition` (the factorisation used here).
