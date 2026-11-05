# Interview notes: Determinants and trace

## Q1. What does a determinant mean, and how would you compute one for a 1000 × 1000 matrix?

**60-second answer.** The determinant of a linear map is the factor by which it scales volume, with
a sign for orientation. A zero determinant means the map squashes some dimension to nothing, so the
matrix is singular. For a large matrix I would not expand cofactors, which costs $n!$ operations.
I would run LU with partial pivoting, which costs $O(n^3)$. I would accumulate the sign from the row
swaps and the sum of $\log|u_{ii}|$ over the pivots, rather than the raw product, so the result does
not overflow or underflow. In practice that is `slogdet`.

**Follow-ups.**
- *Why not multiply the pivots directly?* For unit-variance entries, the product leaves the float64
  range once $n$ passes about 300. The log form stays representable.
- *Is a tiny determinant a sign of trouble?* Not by itself. Scaling a matrix by $c$ multiplies the
  determinant by $c^n$. Check the condition number instead.

## Q2. Why is a small determinant a poor warning of near-singularity?

**60-second answer.** The determinant depends on scale. $0.1\,I_{100}$ has determinant $10^{-100}$,
and it is perfectly conditioned. Near-singularity is about the ratio of the largest to the smallest
singular value, which is the condition number, not the size of the product of singular values. I would
use the condition number, or the smallest singular value relative to the largest, to decide whether a
matrix is nearly singular.

**Follow-ups.**
- *Then what does a zero determinant mean in floating point?* Only that elimination found an exact
  zero pivot. Rounding usually produces a tiny nonzero value instead, so the test needs a tolerance.
- *How does this affect a Gaussian model?* A near-zero covariance determinant can still be valid, so
  the log-likelihood should use $\log\det\Sigma$ computed stably.

## Q3. Where does the log-determinant show up in ML?

**60-second answer.** In the multivariate Gaussian density, the log-likelihood contains
$-\tfrac12\log\det\Sigma$. It penalises spread-out distributions, and it is what makes a
maximum-likelihood fit trade off the covariance size against the fit to the data. It also appears in
Gaussian-process marginal likelihoods and in the evidence for Bayesian model comparison. It is always
computed as a log, through a Cholesky factor, where $\log\det\Sigma = 2\sum_i \log L_{ii}$.

**Follow-ups.**
- *Why Cholesky rather than LU?* $\Sigma$ is symmetric positive definite, so Cholesky is about twice as
  cheap and numerically stable.
- *What is the trace used for here?* Expected quadratic forms: $\mathbb{E}[\mathbf{x}^\top A\mathbf{x}] = \operatorname{tr}(A\Sigma)$ for a zero-mean $\mathbf{x}$ with covariance $\Sigma$.

## Common mistakes

- Computing a determinant with cofactor recursion. It is correct for $2\times2$ and $3\times3$ and
  hopeless beyond that.
- Forgetting the sign of row swaps, so the result has the wrong sign rather than the wrong magnitude.
- Reading a small determinant as a near-singularity alarm, without checking scale.
- Assuming $\operatorname{tr}(AB) = \operatorname{tr}A \cdot \operatorname{tr}B$.
- Returning a raw determinant for a large matrix, which overflows silently into `inf`.
