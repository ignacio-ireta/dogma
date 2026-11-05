## Hint 1: concept

EM alternates two steps. The E-step computes, for each sample, the probability that each
component produced it (the responsibility). The M-step then refits each component by maximum
likelihood, with every sample weighted by its responsibility. Keep one kind of quantity in
mind: the responsibilities, and the effective sample size $N_j$ they add up to.

## Hint 2: formula / approach

The responsibility is $r_{ij} = \pi_j\mathcal{N}_{ij} / \sum_l \pi_l\mathcal{N}_{il}$. The Gaussian density has
a log of $-\tfrac12[d\log 2\pi + \log\det\Sigma + (\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)]$.
Compute the log of the numerator and subtract a row-wise log-sum-exp, so the numbers never
underflow. The M-step uses $N_j = \sum_i r_{ij}$, then $\boldsymbol\mu_j$ from weighted sums, and the
covariance from weighted outer products of $\mathbf{x}_i - \boldsymbol\mu_j$, where $\boldsymbol\mu_j$ is the new mean. The floor
is an eigenvalue operation, not an addition.

## Hint 3: code shape

Write one helper that returns the $(n, k)$ matrix of $\log\pi_j + \log\mathcal{N}_{ij}$, using
`np.linalg.cholesky` and `np.linalg.solve` for each component. Write one E-step helper that
returns the responsibilities and the total log-likelihood. In the loop, record the
log-likelihood before the first iteration, then once after each M-step, so the list has
`n_iter + 1` entries.
