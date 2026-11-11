---
title: Logistic regression
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Data: $n$ samples, features $X \in \mathbb{R}^{n \times d}$ (one row per sample), labels $\mathbf{y} \in \{0, 1\}^n$. The model is

$$\eta_i = b + \mathbf{x}_i^\top \mathbf{w}, \qquad p_i = \sigma(\eta_i) = P(y_i = 1 \mid \mathbf{x}_i).$$

The scalar intercept is $b\in\mathbb R$, the weights are $\mathbf w\in\mathbb R^d$, and each row vector is $\mathbf x_i\in\mathbb R^d$. The Bernoulli likelihood gives the **log-loss** of one sample, $\ell_i = -[y_i \log p_i + (1-y_i)\log(1-p_i)]$. Written in terms of the score it becomes $\ell_i = \log(1 + e^{\eta_i}) - y_i\eta_i$, which avoids forming $\log p_i$ algebraically. Numerically, evaluate the softplus as $\max(\eta_i,0)+\log1p(e^{-|\eta_i|})$ to avoid exponential overflow.

Fit by minimising the penalised mean loss, with an **unpenalised intercept**:

$$J(b, \mathbf{w}) = \frac{1}{n}\sum_i \ell_i + \frac{\lambda}{2}\lVert \mathbf{w} \rVert_2^2.$$

Let $A = [\mathbf{1}, X]\in\mathbb R^{n\times(d+1)}$ and $\boldsymbol\theta = (b, \mathbf{w})\in\mathbb R^{d+1}$, so $\boldsymbol\eta = A\boldsymbol\theta$. The derivatives are

$$\nabla J = \frac{1}{n}A^\top(\mathbf{p} - \mathbf{y}) + \Lambda\boldsymbol\theta, \qquad H = \frac{1}{n}A^\top W A + \Lambda, \qquad W = \operatorname{diag}\big(p_i(1-p_i)\big),$$

where $\lambda\ge0$ is the scalar penalty, $\Lambda = \operatorname{diag}(0, \lambda, \dots, \lambda)\in\mathbb R^{(d+1)\times(d+1)}$, and $W\in\mathbb R^{n\times n}$. The intercept's zero entry in $\Lambda$ is what "unpenalised" means.

## Worked example

$X = (0, 1, 2, 3)^\top$, $\mathbf{y} = (0, 0, 1, 1)^\top$, $\lambda = 1$. The minimiser is $b = -0.5748$, $w = 0.3832$.

| $x$ | $\eta = b + wx$ | $p = \sigma(\eta)$ |
|---|---|---|
| 0 | $-0.575$ | $0.360$ |
| 1 | $-0.192$ | $0.452$ |
| 2 | $0.192$ | $0.548$ |
| 3 | $0.575$ | $0.640$ |

The mean of the four probabilities is $0.5$, the same as the positive rate. That is not a coincidence (see *Why it works*).

Newton from $\boldsymbol\theta = 0$: the largest gradient entry goes $0.50 \to 0.0029 \to 2.7\times10^{-7} \to 2.4\times10^{-15}$. Three steps bring it below $10^{-14}$.

## Why it works

**Convexity.** The Hessian is $\frac1n A^\top W A + \Lambda$. Since $W$ has non-negative entries, the first term is positive semidefinite. With $\lambda > 0$ the penalty makes $H$ positive definite at finite parameters. A finite unique minimiser exists when both classes appear; with only one class, the unpenalised intercept can diverge. See Murphy, Probabilistic Machine Learning: An Introduction, chapter on logistic regression.

**Calibration in the large.** The gradient in the intercept is $\frac1n\sum_i (p_i - y_i)$. At the optimum this is zero, so the mean predicted probability equals the observed positive rate. This holds only because the intercept is unpenalised.

**Separation.** If one hyperplane splits the classes perfectly, the unpenalised likelihood keeps increasing as $\lVert\boldsymbol\theta\rVert \to \infty$, and no finite maximum exists. Data $x = (-2, -1, 1, 2)$, $y = (0, 0, 1, 1)$ show it: $\lambda = 1$ gives $w = 0.472$, $\lambda = 0.1$ gives $w = 1.458$, and $\lambda = 0.01$ gives $w = 2.902$. Shrinking the penalty lets the slope grow.

## Signal vs. noise

- **Signal:** the gradient is $X^\top(\mathbf{p} - \mathbf{y})/n$ plus the penalty term. Each sample pushes the weights by its residual.
- **Signal:** the Hessian weights each sample by $p(1-p)$, so confident samples barely move the fit.
- **Signal:** separation is a property of the data, not a bug in the solver. A penalty is the standard fix.
- **Noise for now:** the choice between Newton, IRLS and quasi-Newton methods. They reach the same optimum.

## Where it breaks

- **Floating-point stalls.** The objective is of order one, so its rounding error is about $10^{-16}$. Near the optimum the predicted decrease from a Newton step can fall below that. A backtracking line search then keeps halving the step, and the gradient stops falling well above the tolerance. The fix is a slack of a few ulps in the sufficient-decrease test.
- **Overflow in the loss.** Writing the loss as $-\log p$ for a saturated $p$ gives $\infty$. Use the stable softplus form $\max(\eta,0)-y\eta+\log1p(e^{-|\eta|})$.
- **Unpenalised separation.** The coefficients grow without bound, and the solver either stops at an arbitrary point or diverges.
- **Scale.** The penalty depends on the feature scale, so standardise features before comparing penalised fits.

**Full Newton steps on real data.** Take the 569 raw breast-cancer rows (column means from 0.004 to 880), label the top 1% of a linear score as positive (6 positives), and use $\ell_2 = 0.001$. The textbook update that always takes the full Newton step drives the linear score to about $2\times10^{6}$ and makes the Hessian singular after 11 steps. The damped solver returns a finite fit whose largest score is about 1900. A naive $\log(1+e^{\eta})$ overflows on the same path, because $e^{\eta}$ is not finite beyond $\eta \approx 709$.

## Interview angle

*60-second answer:* Logistic regression minimises the penalised Bernoulli log-loss. The gradient is $X^\top(\mathbf{p}-\mathbf{y})/n$ plus the penalty term, and the Hessian is $X^\top WX/n$ plus the penalty, so the problem is convex. Newton's method converges in a few steps. Without a penalty, separable data has no finite optimum. With an unpenalised intercept, the fitted probabilities average to the observed rate.

*Likely follow-ups:* Why is the loss convex? (The Hessian is positive semidefinite, because $W$ has non-negative entries.) What does the penalty change under separation? (It makes the minimiser unique and finite.)

## Connections

- **Prerequisites:** `cml/sigmoid-and-log-odds` (the link and its slope), `cml/maximum-likelihood-and-map` (the log-loss as a likelihood), `cml/chain-rule` (the gradient).
- **Unlocks:** `cml/softmax` (the multi-class version), `cml/probability-calibration` (the base-rate property is a first check of calibration).
- **Lineage:** the Bernoulli GLM with a logit link, fitted by iteratively reweighted least squares. Source: Murphy, chapter on logistic regression.
