# Interview notes: Maximum likelihood and MAP

## Q1. Derive the Gaussian maximum-likelihood estimates, and say why the covariance is biased.

**60-second answer.** Write the log-likelihood as a sum over the $n$ observations. Setting the gradient with respect to
$\mu$ to zero gives $\hat\mu = \bar x$. Substituting that in and setting the gradient with respect to $\Sigma$ to zero
gives $\hat\Sigma = \frac1n\sum_i(x_i-\hat\mu)(x_i-\hat\mu)^\top$. The MLE divides by $n$, but $\hat\mu$ was estimated from the
same data, which uses up one degree of freedom. So $\mathbb{E}[\hat\Sigma] = \frac{n-1}{n}\Sigma$. Multiplying by
$\frac{n}{n-1}$ gives the unbiased sample covariance. The bias vanishes as $n$ grows.

**Follow-ups.**
- *Why not use the unbiased estimate for everything?* The MLE is the maximiser of the likelihood, and it has lower
  variance. For large $n$ the difference is negligible.
- *What fails when $n \le d$?* The covariance is singular, so the likelihood is unbounded and the density cannot be
  evaluated.

## Q2. How does MAP differ from MLE, and what does a Gaussian prior do?

**60-second answer.** MLE maximises the log-likelihood. MAP maximises the log-likelihood plus the log-prior, which is
the log of the posterior up to a constant. The prior acts as a penalty in log space. A Gaussian prior on the weights of
a linear model adds $-\lambda\lVert w\rVert^2$, which is ridge regression. A Laplace prior adds $-\lambda\lVert w\rVert_1$,
which is the lasso. As the data grow, the likelihood dominates and MAP approaches MLE.

**Follow-ups.**
- *Is MAP a Bayesian estimate?* It is a point summary of the posterior, the mode. The posterior mean is another
  choice, and it is not the same number.
- *For a Beta-Bernoulli model, what are the pseudo-counts?* $\alpha - 1$ successes and $\beta - 1$ failures are added to
  the observed counts.

## Q3. Why can a MAP estimate sit on the boundary of the parameter space?

**60-second answer.** A density can be unbounded at an endpoint. For a $\mathrm{Beta}(a,b)$ with $a < 1$ the density
diverges at 0, so the mode is 0 and the formula $(a-1)/(a+b-2)$ gives a meaningless value. The correct convention
returns the boundary. Check whether the mode is interior before you use the closed form. For the coin model with at
least one flip, the boundary can only occur with no heads and $\alpha \le 1$, or with all heads and $\beta \le 1$.

**Follow-ups.**
- *Why does a uniform prior make MAP equal MLE?* Its log-density is constant, so it adds nothing to the maximisation.
- *What happens to MAP at $n=0$?* The estimate is just the prior mode, which may not exist when both parameters are
  below 1.

## Common mistakes

- Calling a likelihood a probability. It is not a distribution over $\theta$ and need not sum or integrate to 1.
- Reporting the MAP as the posterior mean, or the reverse.
- Using the $n-1$ divisor in the MLE and then comparing its likelihood to the MLE's.
- Forgetting that a density can exceed 1, so a large likelihood does not mean a high probability.
