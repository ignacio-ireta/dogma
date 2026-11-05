---
title: Maximum likelihood and MAP
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A model has parameters $\theta$ and data $\mathcal{D}$. Two functions share the same formula but answer
different questions:

- **Probability** $p(\mathcal{D}\mid\theta)$: fix $\theta$, vary the data. It describes what you would see.
- **Likelihood** $L(\theta) = p(\mathcal{D}\mid\theta)$: fix the data, vary $\theta$. It scores parameters.

The **log-likelihood** $\ell(\theta) = \log L(\theta)$ turns products over independent observations into sums and
has the same maximiser. The sum form assumes the rows are independent and identically distributed given
$\theta$. For dependent data, such as a time series, $L$ is not a product over rows, and the row formulas
below do not apply. The **maximum-likelihood estimate** (MLE) is $\hat\theta = \arg\max_\theta \ell(\theta)$.
The **maximum-a-posteriori** (MAP) estimate adds a prior $p(\theta)$:
$$\hat\theta_{\text{MAP}} = \arg\max_\theta \big[\ell(\theta) + \log p(\theta)\big].$$

MAP is the posterior mode, a point summary; the posterior is still a distribution. Per the Library note on
Bishop (2006), estimation, latent inference and prediction are separate computations, and prediction
averages over the posterior rather than plugging in the mode
([source: library-refresh/02-AI-ML/00-Overviews-and-Books/2006 -- Bishop -- Probabilistic Models, Latent Variables and Approximate Inference -- Study Note.md]).

| Model | MLE | MAP |
|---|---|---|
| Gaussian $\mathcal{N}(\mu,\Sigma)$, $n$ rows | $\hat\mu = \frac1n\sum_i x_i$, $\ \hat\Sigma = \frac1n\sum_i (x_i-\hat\mu)(x_i-\hat\mu)^\top$ | not in this topic |
| Coin, $k$ heads in $n$ flips, $\mathrm{Beta}(\alpha,\beta)$ prior | $k/n$ | mode of $\mathrm{Beta}(\alpha+k,\ \beta+n-k)$ |

## Worked example

**Gaussian.** The three points $(1,2),(3,4),(5,9)$ have mean $(3,5)$. Deviations: $(-2,-3),(0,-1),(2,4)$.
The MLE covariance divides the summed products by $n=3$:
$\hat\Sigma = \begin{pmatrix}8/3 & 14/3\\ 14/3 & 26/3\end{pmatrix} \approx \begin{pmatrix}2.67&4.67\\4.67&8.67\end{pmatrix}$.
The unbiased sample covariance divides by $n-1=2$ and gives $\begin{pmatrix}4&7\\7&13\end{pmatrix}$.

**Beta-Bernoulli.** Prior $\mathrm{Beta}(2,2)$, data $k=3$ heads in $n=10$ flips.

1. MLE: $k/n = 0.3$.
2. Posterior: $\mathrm{Beta}(a,b)$ with $a = 2+3 = 5$ and $b = 2+7 = 9$.
3. MAP is the mode: $(a-1)/(a+b-2) = 4/12 = 0.333$. The posterior mean would be $a/(a+b) = 5/14 = 0.357$.
   Use the mode for MAP, the mean for a Bayes estimate.

## Why it works

**Gaussian MLE.** Write $\ell(\mu,\Sigma)$ as a sum over rows. Setting $\partial\ell/\partial\mu = 0$ gives
$\hat\mu = \bar x$. Substituting back and setting the derivative with respect to $\Sigma$ to zero gives the average of
the outer products, with divisor $n$ (Bishop, *Pattern Recognition and Machine Learning*, chapter on probability
distributions). Dividing by $n$ is what maximises the likelihood. It is biased: $\mathbb{E}[\hat\Sigma] = \frac{n-1}{n}\Sigma$.
Simulation confirms it: for each size, 200,000 standard normal replicates seeded with `default_rng(0)`
give a mean one-dimensional MLE variance of $0.503$ at $n=2$ (target $0.5$) and $0.802$ at $n=5$ (target $0.8$).
Multiplying by $n/(n-1)$ gives the unbiased estimate. On the five points above plus $(2,2)$ and $(4,7)$, the
log-likelihood is $-14.38$ at the MLE and $-14.50$ at the unbiased covariance.

**MAP.** The posterior is proportional to likelihood times prior. For a Beta prior, $\log p(\theta) = (\alpha-1)\log\theta
+ (\beta-1)\log(1-\theta) + \text{const}$, so the prior adds $\alpha-1$ pseudo-successes and $\beta-1$ pseudo-failures:
$\hat\theta_{\text{MAP}} = \frac{k+\alpha-1}{n+\alpha+\beta-2}$. A Gaussian prior on regression weights adds
$-\lambda\lVert w\rVert^2$ to the log-posterior, which is L2 regularisation (ridge). A Laplace prior gives L1 (lasso).

**Boundaries.** With $a=\alpha+k$ and $b=\beta+n-k$, the interior mode needs $a > 1$ and $b > 1$.
Otherwise $a \le 1$ forces $k=0$, where the density is largest at 0, and $b \le 1$ forces $k=n$,
where it is largest at 1. The p02 convention follows from this.

**Learning as compression.** Per the Library note on MacKay (2003), maximising the likelihood of i.i.d. data
minimises the total log loss, or code length, of the data under the model. That is a coding reading of MLE,
not a generalisation guarantee
([source: library-refresh/01-Mathematics/07-Information-Theory/2003 -- MacKay -- Coding, Bayesian Inference and Learning as Compression -- Study Note.md]).

## Signal vs. noise

- **Signal:** likelihood versus probability, and the log-likelihood as the object to maximise.
- **Signal:** MAP $=$ MLE plus log-prior. A prior is a penalty in log space.
- **Signal:** MAP converges to MLE as data grow, at rate $O(1/n)$ for a fixed prior. With $\mathrm{Beta}(2,2)$ and
  $k/n = 0.3$, the gap is $0.4/(n+2)$: $0.033$ at $n=10$ and $0.0004$ at $n=1000$.
- **Noise for now:** full Bayesian inference, conjugacy beyond Beta-Bernoulli, and the evidence.

## Where it breaks

- **Bias.** The Gaussian MLE covariance underestimates the spread by $(n-1)/n$. Small $n$ makes this large.
- **Singular estimates.** With $n \le d$ the MLE covariance has rank at most $n-1$. Its log-likelihood is
  unbounded (the covariance can shrink in the unobserved directions), so inversion and density evaluation fail.
- **Density is not probability.** A continuous density can exceed 1, and a likelihood can be unbounded. The MLE of a
  mixture of Gaussians is unbounded when one component collapses onto a single point.
- **Boundary MAP.** MAP sits on the boundary only when the posterior shape $a=\alpha+k\le1$ (no heads) or
  $b=\beta+n-k\le1$ (all heads). A prior with $\alpha<1$ alone is not enough. Any $k\ge1$ lifts $a$ above 1,
  but $b\le1$ can still hold, as with $\alpha=\beta=0.5$, $k=n=1$. For $\alpha=0.5$, $k=1$, $n=1$, $\beta=2$ the
  interior mode is $1/3$. A naive formula $(a-1)/(a+b-2)$ applied at $a\le1$ gives a meaningless value, such
  as $1$ at $a=0.5$, $b=1$.

## Interview angle

"Derive the Gaussian MLE, and say why the covariance is biased." Answer shape: write the log-likelihood, set the
gradients to zero, get the mean and the average outer product, then state $\mathbb{E}[\hat\Sigma] = \frac{n-1}{n}\Sigma$
and divide by $n-1$ when you need an unbiased estimate. Follow-up: what a Gaussian prior does (L2 regularisation). Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/normal-and-multivariate-gaussian` (the likelihood p01 maximises), `cml/bernoulli-binomial-and-poisson`
  (the likelihood p02 maximises).
- **Unlocks:** `cml/ridge-regression-and-weight-decay` (MAP with a Gaussian prior), `cml/lasso-and-elastic-net` (MAP with a
  Laplace prior), and every model fitted by maximising a log-likelihood.
