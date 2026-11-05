---
title: Gaussian mixtures and EM
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A Gaussian mixture models each sample as drawn from one of $k$ hidden components. With samples
$\mathbf{x}\in\mathbb{R}^d$:

$$p(\mathbf{x}) = \sum_{j=1}^{k}\pi_j\,\mathcal{N}(\mathbf{x}\mid\boldsymbol\mu_j,\Sigma_j), \qquad \sum_j \pi_j = 1.$$

The label $z_i\in\{1,\dots,k\}$ of sample $i$ is hidden. EM works with the **responsibility**
$r_{ij} = P(z_i = j \mid \mathbf{x}_i)$, a number in $[0,1]$ with $\sum_j r_{ij} = 1$.

| Step | Formula | Shape | Meaning |
|---|---|---|---|
| E-step | $r_{ij} = \dfrac{\pi_j\mathcal{N}_{ij}}{\sum_l \pi_l\mathcal{N}_{il}}$ | $(n,k)$ | Soft assignment |
| M-step: mass | $N_j = \sum_i r_{ij}$ | $(k)$ | Effective sample size |
| M-step: weight | $\pi_j = N_j / n$ | $(k)$ | Share of the data |
| M-step: mean | $\boldsymbol\mu_j = \frac{1}{N_j}\sum_i r_{ij}\,\mathbf{x}_i$ | $(k,d)$ | Weighted mean |
| M-step: covariance | $\Sigma_j = \frac{1}{N_j}\sum_i r_{ij}(\mathbf{x}_i-\boldsymbol\mu_j)(\mathbf{x}_i-\boldsymbol\mu_j)^\top$ | $(k,d,d)$ | Weighted scatter about the new mean |

The M-step is maximum likelihood with each sample weighted by $r_{ij}$. Hard-assignment $k$-means is
the limit where every $r_{ij}\in\{0,1\}$ and every covariance is the same spherical matrix.

## Worked example

Two samples $x = 0$ and $x = 2$, $k = 2$, equal weights, means initialised at $0$ and $2$, and
covariances equal to the pooled variance, which is $1$.

1. E-step: $\mathcal{N}(0\mid 0,1)$ is proportional to $1$ and $\mathcal{N}(0\mid 2,1)$ to $e^{-2}$. So
   $r_{0,1} = \frac{1}{1+e^{-2}} = 0.8808$ and $r_{2,1} = 0.1192$. By symmetry $r_{0,2} = 0.1192$ and $r_{2,2} = 0.8808$.
2. Masses: $N_1 = N_2 = 1$, so the weights stay $0.5$.
3. Means: $\mu_1 = 0.1192\cdot 2 = 0.2384$ and $\mu_2 = 0.8808\cdot 2 = 1.7616$. Each mean moves
   toward the sample it is responsible for.
4. Covariances: both equal $4r(1-r) = 0.4200$.
5. Log-likelihood: $\ell$ rises from $-2.9703$ to $-2.4394$ after one iteration.

## Why it works

For any distribution $q_{ij}$ over the label of sample $i$, Jensen's inequality gives

$$\ell(\theta) = \sum_i\log\sum_j \pi_j\mathcal{N}_{ij} \;\ge\; \sum_i\sum_j q_{ij}\log\frac{\pi_j\mathcal{N}_{ij}}{q_{ij}} =: F(q,\theta),$$

with equality exactly when $q_{ij} = r_{ij}(\theta)$. The E-step sets $q$ to the responsibilities, so
the bound is tight at the current $\theta_t$. The M-step maximises $F$ over $\theta$, which is the
weighted maximum likelihood above. Hence $\ell(\theta_{t+1}) \ge F(q_t,\theta_{t+1}) \ge F(q_t,\theta_t) = \ell(\theta_t)$.
This is the monotonicity result of Dempster, Laird & Rubin (1977), and the mixture case is in Bishop
(*PRML*, ch. 9).

The covariance floor must be a constrained M-step as well. Maximising $F$ over covariances whose
eigenvalues are all at least $\varepsilon$ gives the eigenvalue clip $\max(\lambda,\varepsilon)$, and the
chain above still holds. Adding $\varepsilon I$ after the update is not the constrained maximiser, so the bound need not rise. On
the two-point fixture with $\varepsilon = 0.1$, one iteration gives $\ell = -2.575$, against $-2.439$ for the
eigenvalue clip. On random 2-D data with spread about $0.03$ and $\varepsilon = 10^{-3}$, one-iteration drops
of several nats occurred in our check, while the clip never dropped by more than rounding (about
$10^{-11}$). The chain also needs the previous parameters to be feasible, which the floored initial state
guarantees.

## Signal vs. noise

- **Signal:** the responsibilities, the weighted M-step, and the fact that $\ell$ never decreases.
- **Signal:** shapes. Responsibilities are $(n,k)$, covariances are $(k,d,d)$, and every sum runs
  over the sample axis.
- **Noise for now:** full versus diagonal covariance trade-offs, convergence tolerances, and choosing
  $k$ (BIC and held-out likelihood come later).

## Where it breaks

- **Collapse.** If one component's responsibilities concentrate on one point or on duplicate points,
  its scatter is zero and its density spikes without bound. Floor every eigenvalue at $\varepsilon$.
- **Underflow.** For samples far from every component, $\mathcal{N}$ is $e^{-10^8}=0$ in float64, which
  gives $0/0$. Compute log-densities and normalise with log-sum-exp.
- **Initialisation.** EM finds a local maximum. On the nine points $0, 0.5, 1, 4, 4.5, 5, 9, 9.5, 10$ with $k=2$, run from the initial state of p01
  (equal weights, pooled covariance) for about 100 iterations, starting means $0$ and $5/3$ converge to
  means $0.49$ and $6.91$ with $\ell = -21.31$. Starting means $0$ and $25/3$ converge to $2.51$ and
  $9.50$ with $\ell = -20.09$. Use several
  starts and keep the best $\ell$.
- **Dead components.** A component with $N_j = 0$ cannot be refitted. Do not divide by $N_j$. Keep
  it with weight $0$.

## Interview angle

Sixty seconds: the E-step computes responsibilities in log space, the M-step runs weighted maximum
likelihood, and the Jensen bound makes the log-likelihood rise every iteration. EM converges to a
local maximum, so the start matters. Covariances can collapse, and a floor fixes that. $k$-means is
the hard-assignment limit. Follow-ups: why the likelihood can't decrease (the tight bound), and what
the floor changes (the constrained maximiser, not a penalty). See `interview.md`.

## Connections

- **Prerequisites:** `cml/k-means` (the hard limit), `cml/normal-and-multivariate-gaussian` (the density),
  `cml/maximum-likelihood-and-map` (the weighted MLE).
- **Unlocks:** model selection with information criteria, and mixture-based anomaly scores
  (`cml/anomaly-scoring`).
- **Lineage:** Dempster, Laird & Rubin (1977) formalised EM with the latent-variable bound. Bishop
  (2006) gives the Gaussian mixture treatment in ch. 9, and Deisenroth, Faisal & Ong (2020) cover
  mixture density estimation in their book, ch. 11.
