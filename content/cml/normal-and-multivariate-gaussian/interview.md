# Interview notes: Normal and multivariate Gaussian

## Q1. Write the log-density of a multivariate normal and compute it stably.

**60-second answer.** $\log\mathcal{N}(\mathbf{x}\mid\boldsymbol\mu,\Sigma) = -\tfrac{d}{2}\log 2\pi -
\tfrac12\log\det\Sigma - \tfrac12(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)$. In code,
factor $\Sigma = LL^\top$ with Cholesky. The factorisation fails exactly when $\Sigma$ is not positive definite,
which doubles as the validity check. The log-determinant is $2\sum_i\log L_{ii}$. The quadratic form is
$\lVert\mathbf{z}\rVert^2$ where $L\mathbf{z} = \mathbf{x}-\boldsymbol\mu$. I never form $\Sigma^{-1}$ or $\det\Sigma$:
the first is slower and less accurate, and the second overflows or underflows in high dimension.

**Follow-ups.**
- *Why not `np.log(np.linalg.det(cov))`?* At dimension 60 with eigenvalues near $10^{-7}$, the determinant is
  $0$ and its logarithm is $-\infty$.
- *What must you check before factoring?* Symmetry. Cholesky reads one triangle, so an asymmetric input gives a
  wrong answer without an error.

## Q2. What is the Mahalanobis distance, and when is it better than Euclidean distance?

**60-second answer.** It is $\sqrt{(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)}$. It rescales
each direction by the spread of the distribution along it, so it accounts for correlation and unequal variances.
Euclidean distance treats every direction alike, so a point 2 units away along a high-variance direction can look
farther than one 1 unit away along a low-variance direction, when the Mahalanobis distance says the reverse. It is
the natural outlier score for roughly Gaussian data, and it is the quantity whose level sets are the density's
ellipsoids.

**Follow-ups.**
- *What is its distribution under a Gaussian?* $\delta^2$ is chi-squared with $d$ degrees of freedom. That gives a
  principled cutoff for outliers.
- *What breaks it?* A singular or badly estimated $\Sigma$. With $n<d$ the sample covariance is not invertible.

## Q3. Given a joint Gaussian, what are the marginal and conditional distributions?

**60-second answer.** Both are Gaussian. The marginal of a block takes the matching block of the mean and of the
covariance. The conditional of $\mathbf{x}_1$ given $\mathbf{x}_2$ has mean $\boldsymbol\mu_1 + \Sigma_{12}\Sigma_{22}^{-1}
(\mathbf{x}_2-\boldsymbol\mu_2)$ and covariance $\Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$. The conditional
mean is linear in the observed value, and the conditional covariance does not depend on it. That is why Gaussian
models give closed-form updates, as in Kalman filtering and Gaussian process regression.

**Follow-ups.**
- *Is the conditional covariance ever larger than the marginal?* No. Conditioning on data never increases
  uncertainty in expectation, and for a Gaussian the conditional covariance is always no larger.
- *Why is a linear combination of Gaussian variables Gaussian?* Because the joint vector is Gaussian, and a linear
  map of a Gaussian vector is Gaussian.

## Common mistakes

- Dropping the $-\tfrac{d}{2}\log 2\pi$ constant when only comparing densities. Harmless for ranking, wrong for
  reporting a log-likelihood.
- Taking the log of a density that has already underflowed to zero.
- Using $\Sigma^{-1}$ explicitly, then wondering why results disagree for ill-conditioned covariances.
- Assuming a Gaussian fits because the marginals look bell-shaped. Joint tails can still be heavy.
