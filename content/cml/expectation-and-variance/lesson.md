---
title: Expectation and variance
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A discrete random variable $X$ takes values $x_1,\dots,x_n$ with probabilities $p_1,\dots,p_n$, where
$p_i \ge 0$ and $\sum_i p_i = 1$. Every quantity below is a weighted average over those $n$ outcomes.

| Quantity | Formula | Meaning |
|---|---|---|
| Mean (expectation) | $\mu = \mathbb{E}[X] = \sum_i p_i x_i$ | the long-run average |
| Variance | $\operatorname{Var}(X) = \mathbb{E}[(X-\mu)^2] = \sum_i p_i (x_i-\mu)^2$ | average squared distance from the mean |
| Standard deviation | $\sigma = \sqrt{\operatorname{Var}(X)}$ | same units as $X$ |
| Second moment | $\mathbb{E}[X^2] = \sum_i p_i x_i^2$ | an intermediate, not the variance |

The shortcut $\operatorname{Var}(X) = \mathbb{E}[X^2] - \mu^2$ is algebraically identical to the
definition. It is the formula you should not compute with (see *Where it breaks*).

Four rules cover almost everything:

- **Linearity:** $\mathbb{E}[aX + bY + c] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y] + c$. Dependence does not
  matter, if the expectations are finite.
- **Affine maps:** $\operatorname{Var}(aX+b) = a^2\operatorname{Var}(X)$. The shift $b$ drops out and the
  slope is squared.
- **Sums:** $\operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X,Y)$,
  where $\operatorname{Cov}(X,Y) = \mathbb{E}[(X-\mathbb{E}X)(Y-\mathbb{E}Y)]$. Variances add only when the
  covariance is zero.
- **Total variance:** $\operatorname{Var}(X) = \mathbb{E}[\operatorname{Var}(X\mid Z)] +
  \operatorname{Var}(\mathbb{E}[X\mid Z])$. Average within-group spread plus between-group spread.

## Worked example

$X$ takes the values $\{0, 1, 2, 10\}$ with probabilities $\{0.5, 0.25, 0.2, 0.05\}$.

1. **Mean.** $0(0.5) + 1(0.25) + 2(0.2) + 10(0.05) = 1.15$.
2. **Centred squares.** Subtract $\mu = 1.15$, square, then weight by $p$:

| $x$ | $p$ | $x-\mu$ | $p\,(x-\mu)^2$ |
|---|---|---|---|
| 0 | 0.50 | $-1.15$ | 0.66125 |
| 1 | 0.25 | $-0.15$ | 0.005625 |
| 2 | 0.20 | $0.85$ | 0.1445 |
| 10 | 0.05 | $8.85$ | 3.916125 |

3. **Variance** = sum of the last column $= 4.7275$, and $\sigma = 2.174$.
4. **Check with the shortcut.** $\mathbb{E}[X^2] = 0 + 0.25 + 0.8 + 5.0 = 6.05$, and
   $6.05 - 1.15^2 = 4.7275$.

Apply the affine rule to $Y = 2X+1$: $\mathbb{E}[Y] = 2(1.15)+1 = 3.3$ and
$\operatorname{Var}(Y) = 4(4.7275) = 18.91$.

Dependence matters for sums. With $Y=X$ the formula gives $4\operatorname{Var}(X)$, which is right
because $X+X=2X$. Two independent draws would give $2\operatorname{Var}(X)$.

## Why it works

Linearity comes from summing over the joint distribution, so it needs only finite expectations. Variance is
the second moment of the centred variable. Expanding the square gives the shortcut:
$\mathbb{E}[(X-\mu)^2] = \mathbb{E}[X^2] - 2\mu\,\mathbb{E}[X] + \mu^2 = \mathbb{E}[X^2] - \mu^2$.

Expanding $\operatorname{Var}(X+Y)$ gives the covariance term. Total variance comes from splitting
$X - \mu = (X - \mathbb{E}[X\mid Z]) + (\mathbb{E}[X\mid Z] - \mu)$. The cross term has expectation zero
after conditioning on $Z$, so the two parts add. Example: a fair coin $Z$ chooses between $\{0,2\}$
(when $Z=0$) and $\{4,6\}$ (when $Z=1$), each value equally likely. Then $\mathbb{E}[X]=3$, the
within-group variance is 1 in both groups, and the group means 1 and 5 have variance 4. Total: $1+4=5$.
Direct computation on $\{0,2,4,6\}$ gives $\mathbb{E}[X^2]=14$ and $14-9=5$. (Blitzstein & Hwang,
*Introduction to Probability*, chapter on expectation.)

**Moving a gradient inside an expectation.** The score-function identity $\nabla_\theta\mathbb{E}_{p_\theta}[f]=\mathbb{E}[f\,\nabla_\theta\log p_\theta]$ needs $p_\theta$ to be differentiable in $\theta$, with an integrable envelope for $f\,\nabla_\theta p_\theta$ near $\theta$. It does not need $f$ to be differentiable, and it works for discrete $X$. The reparameterization $X=g(\theta,\varepsilon)$ instead needs $f\circ g$ to be differentiable in $\theta$ with an integrable envelope, and it has no discrete form without a continuous relaxation. The note supports only the dominated-convergence step (Axler §3B, theorem 3.31, locator not checked); the other conditions are standard results. [source: library-refresh/01-Mathematics/04-Analysis-and-Measure/2026 -- Library -- Dominated Convergence and Gradients of Expectations -- Concept Note.md].

## Signal vs. noise

- **Signal:** linearity; the affine rule; the covariance term in $\operatorname{Var}(X+Y)$; centring
  before squaring.
- **Signal:** probabilities sum to 1; a set summing to 0.9 is a bug.
- **Noise for now:** moment generating functions, skewness and kurtosis, and continuous densities.

## Where it breaks

- **Cancellation.** Take $X \in \{0,1,2\}+10^9$ with equal probabilities; the true variance is $2/3$.
  Computed as $\mathbb{E}[X^2]-\mu^2$ in float64, both terms sit near $10^{18}$, where the spacing is 128.
  With `p @ x**2 - (p @ x)**2` the result is $-128$, and another summation order gives $0$. Subtracting
  the mean first gives $2/3$, because each $x-\mu$ is small and exact.
- **Heavy tails.** Some distributions have no finite variance. A Pareto distribution with shape
  $\alpha \le 2$ has infinite variance, and a Cauchy distribution has no mean.
- **Rounding in probabilities.** Check the sum with a tolerance, never with $==$: $0.7, 0.1, 0.1, 0.1$
  sum to $0.9999999999999999$ in `np.sum`, which is a valid distribution.
- **Products.** Independence implies zero covariance, not the reverse. $\mathbb{E}[XY]=\mathbb{E}[X]\,\mathbb{E}[Y]$
  holds exactly when $\operatorname{Cov}(X,Y)=0$ (finite second moments assumed), which is weaker than independence.
- **Derivatives and limits.** $\mathbb{E}$ commutes with a derivative only under a condition. Let
  $Z\sim\mathrm{Uniform}(0,1)$ and $R(\theta)=\mathbb{E}[\mathbf 1\{Z\le\theta\}]$. The true $R'(\theta)=1$ on
  $(0,1)$, but the indicator's derivative is 0 almost everywhere, so moving it inside gives 0. Dominated
  convergence allows the move when the difference quotients share an integrable envelope; a step loss
  has none.

## Interview angle

"What is the variance of $X+Y$?" Answer shape: linearity first, which needs only finite means. Then expand
the square: the cross term is the covariance, and independence (or zero correlation) removes it.
Follow-ups: why $\operatorname{Var}(X-Y)$ also adds the variances, and how to compute a variance near
$10^9$ (centre first, or Welford's update). Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/probability-basics`.
- **Unlocks:** `cml/covariance-and-correlation` (the covariance is the cross term above),
  `cml/bernoulli-binomial-and-poisson` (a Bernoulli variable has variance $p(1-p)$),
  `cml/normal-and-multivariate-gaussian`, and `cml/sampling-distributions-standard-errors-and-cis`
  (the variance of a sample mean is $\sigma^2/n$ for independent draws).
