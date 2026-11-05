---
title: Sigmoid and log-odds
minutes: 12
min_viable: "Read The system and the Worked example, then do p01."
---

## The system

For an event with probability $p \in (0, 1)$, the **odds** are $p/(1-p)$ and the **log-odds**, or **logit**, are

$$\operatorname{logit}(p) = \log\frac{p}{1-p}.$$

The logit maps $(0, 1)$ onto the whole real line. Its inverse is the **sigmoid** (logistic function), with $z \in \mathbb{R}$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma(\operatorname{logit}(p)) = p, \qquad \operatorname{logit}(\sigma(z)) = z.$$

Four properties do most of the work:

| Property | Statement | Use |
|---|---|---|
| Midpoint | $\sigma(0) = \tfrac12$ | the decision boundary of a 0.5 threshold |
| Symmetry | $\sigma(-z) = 1 - \sigma(z)$ | computing the small side safely |
| Slope | $\sigma'(z) = \sigma(z)\,(1-\sigma(z)) \le \tfrac14$ | the chain-rule factor in gradients |
| Saturation | $\sigma(z) \to 1$ as $z \to +\infty$, $\to 0$ as $z \to -\infty$ | confident predictions |

**Logistic regression** models the log-odds as linear in the features. For one sample with features $\mathbf{x} \in \mathbb{R}^d$, weights $\boldsymbol\beta$ and intercept $\beta_0$:

$$\log\frac{P(Y=1 \mid \mathbf{x})}{P(Y=0 \mid \mathbf{x})} = \beta_0 + \mathbf{x}^\top\boldsymbol\beta, \qquad P(Y=1 \mid \mathbf{x}) = \sigma(\beta_0 + \mathbf{x}^\top\boldsymbol\beta).$$

The model is linear in the log-odds, not in the probability. A coefficient $\beta_j$ is the change in log-odds for a one-unit increase in $x_j$, with the other features fixed. Its exponential, $e^{\beta_j}$, is the **odds ratio** for that increase.

## Worked example

Three logits, $z = -2, 0, 2$:

| $z$ | $\sigma(z)$ | Check |
|---|---|---|
| $-2$ | $0.1192$ | $e^{-2} = 0.1353$, so $1/(1 + 0.1353) = 0.8808$ and $1 - 0.8808 = 0.1192$ |
| $0$ | $0.5$ | exactly, because $e^0 = 1$ |
| $2$ | $0.8808$ | symmetry: $1 - 0.1192$ |

Odds and log-odds: at $p = 0.8$ the odds are $4$, so $\operatorname{logit}(0.8) = \log 4 = 1.386$.

Odds ratio: if $\beta_1 = \log 2$, then $e^{\beta_1} = 2$. Each one-unit increase in $x_1$ doubles the odds of $Y = 1$. It does not double the probability. For rare outcomes the odds ratio approximates the probability ratio. At $p=0.5$, doubling the odds moves the probability only to $2/3$, not to $1$.

## Why it works

The naive sigmoid $1/(1 + e^{-z})$ fails on one side. For $z = -1000$ the exponent is $+1000$, and $\exp(710)$ already overflows float64. NumPy returns the right value $0$ and emits an overflow warning. The mirror formula $e^{z}/(1+e^{z})$ fails at $z = +710$.

The fix is one exponential that can never be large: $e = e^{-|z|} \in [0, 1]$. Then

$$\sigma(z) = \frac{1}{1+e} \text{ if } z \ge 0, \qquad \sigma(z) = \frac{e}{1+e} \text{ if } z < 0.$$

The second form matters for relative precision. At $z = 30$ the true small value is $\sigma(-30) = 9.3576\times10^{-14}$, while the float computation $1 - \sigma(30)$ gives $9.3481\times10^{-14}$, an error of about $0.1\%$. At $z = -36$ the form $1 - 1/(1+e)$ is off by $4\%$. At $z = -40$ it returns $0$.

## Signal vs. noise

- **Signal:** the logit and the sigmoid are inverses. Log-odds are linear in the features.
- **Signal:** $\beta_j$ is a log odds ratio. $e^{\beta_j}$ is an odds ratio, not a risk ratio.
- **Signal:** $\sigma' = \sigma(1-\sigma)$ is at most $1/4$. This slope is what the gradient multiplies by.
- **Noise for now:** the historical use of the logistic curve in population growth models.

## Where it breaks

- **Overflow in the unused branch.** `np.where` evaluates both branches for every element. A naive exponential in either branch warns, even when the result is correct.
- **Clipping hides the answer.** Clipping the logit at $\pm 500$ changes $\sigma(-700)$. Clipping a probability at $10^{-12}$ turns $\operatorname{logit}(10^{-300})$ into $\operatorname{logit}(10^{-12}) \approx -27.6$.
- **Relative error near $\tfrac12$ for the logit.** The naive $\log\big(p/(1-p)\big)$ has an absolute error near $10^{-16}$. Near $p = \tfrac12$ the answer is close to $0$, so its relative error is large. At $p = \tfrac12 - 1.3\times10^{-11}$ the naive form is off by about $2\times10^{-6}$ in relative terms. Writing it as $\operatorname{log1p}\big((2p-1)/(1-p)\big)$ fixes this, because $2p-1$ is exact near $\tfrac12$.
- **Odds ratio misread as risk ratio.** For a common outcome the two differ a lot.

## Interview angle

*60-second answer:* The sigmoid maps a real score to a probability. Logistic regression says the log-odds are linear in the features, so each coefficient is a log odds ratio, and $e^{\beta_j}$ is the multiplicative change in odds for a unit increase. Numerically, compute $\sigma$ from $e^{-|z|}$, so no exponent is positive, and keep the small tail as its own quantity.

*Likely follow-ups:* Why not a linear probability model? (It can predict outside $[0, 1]$.) Why is $e^{\beta}$ an odds ratio and not a risk ratio? (The model is linear in the log-odds, and the ratio of probabilities is not constant across the feature range.)

## Connections

- **Prerequisites:** `cml/probability-basics` (odds and events), `cml/chain-rule` (the slope $\sigma'$).
- **Unlocks:** `cml/logistic-regression` (the fit), `cml/softmax` (the sigmoid is the two-class softmax), `cml/cross-entropy` (the binary loss with logits).
- **Lineage:** the logit is the canonical link of the Bernoulli member of the generalised linear models. The logistic function was adopted for binary regression in the twentieth century. Source: Murphy, chapter on logistic regression.
