---
title: XGBoost second-order objective
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A boosted model is a sum of small trees, $F(\mathbf{x}) = \sum_t \eta\, f_t(\mathbf{x})$, where $\eta$ is the shrinkage. Sample $i$ has label $y_i$ and current margin $\hat m_i$. The loss gives two numbers per sample:

$$g_i = \frac{\partial \ell}{\partial m}\Big|_{\hat m_i}, \qquad h_i = \frac{\partial^2 \ell}{\partial m^2}\Big|_{\hat m_i}.$$

For the logistic loss, $g_i = \sigma(\hat m_i) - y_i$ and $h_i = \sigma(\hat m_i)(1 - \sigma(\hat m_i))$.

The second-order Taylor expansion around $\hat m$, for a margin change $\delta$, is $\ell(\hat m + \delta) \approx \ell(\hat m) + g\,\delta + \tfrac12 h\,\delta^2$.

A leaf holds samples $I$, with $G = \sum_I g_i$ and $H = \sum_I h_i$. Give it weight $w$. With L2 penalty $\lambda$ and cost $\gamma$ per leaf, the leaf adds $Gw + \tfrac12(H+\lambda)w^2 + \gamma$ to the objective. Its minimiser and minimum are

$$w^* = -\frac{G}{H+\lambda}, \qquad \text{value} = -\tfrac12\frac{G^2}{H+\lambda} + \gamma.$$

Splitting a leaf into $L$ and $R$ lowers the objective by the gain

$$\text{Gain} = \tfrac12\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma.$$

The tree keeps a split only when this gain is positive.

## Worked example

Three samples, labels $y = (1, 0, 1)$, all at margin $0$. So $\sigma = 0.5$, $g = (-0.5, 0.5, -0.5)$ and $h = (0.25, 0.25, 0.25)$. Use $\lambda = 1$ and $\gamma = 0.1$.

| Split | $G_L, H_L$ | $G_R, H_R$ | $w_L$ | $w_R$ | Gain |
|---|---|---|---|---|---|
| $\{1,3\}$ vs $\{2\}$ | $-1.0,\ 0.5$ | $0.5,\ 0.25$ | $0.667$ | $-0.400$ | $0.262$ |
| $\{1,2\}$ vs $\{3\}$ | $0.0,\ 0.5$ | $-0.5,\ 0.25$ | $0.000$ | $0.400$ | $-0.071$ |

The first split is kept. The second scores $+0.029$ without $\gamma$ and $-0.071$ with it, so the leaf cost changes the decision.

The Taylor model for one sample ($y = 1$, margin $0$, $g = -0.5$, $h = 0.25$) against the exact loss $\log(1+e^{\delta}) - \delta$:

| Step $\delta$ | Exact | First order | Second order |
|---|---|---|---|
| $1$ | $0.313$ | $0.193$ | $0.318$ |
| $2$ | $0.127$ | $-0.307$ | $0.193$ |

At $\delta = 2$ the first-order model predicts a negative loss. The second-order model does not.

For $G = -3$, $H = 2$, the weight shrinks as $\lambda$ grows: $1.500$ at $\lambda = 0$, $1.000$ at $1$, $0.429$ at $5$, $0.136$ at $20$.

## Why it works

**Why second order.** The first derivative gives a direction and the second adds curvature. For one sample with no penalty, the minimiser of the quadratic model is the Newton step $-g/h$. See Chen & Guestrin 2016 [library:AI-ML/05-Classification/Chen Guestrin XGBoost 2016.md].

**One algorithm for any loss.** The tree search reads only sums of $g$ and $h$, so a twice-differentiable loss can supply them, but the leaf quadratic needs $H+\lambda>0$ to have the stated minimum; non-convex losses need additional care. Friedman's view: boosting is gradient descent in function space [library:AI-ML/05-Classification/Friedman Gradient Boosting 2001.md].

**Regularisers.**
- $\lambda$ shrinks leaf weights and keeps $H+\lambda > 0$, so a zero-hessian leaf still has a defined weight.
- $\gamma$ is a fixed cost per leaf. It prunes splits whose gain is not positive.
- `min_child_weight` is a minimum on each child's $H$. With logistic loss $h_i \le 0.25$, so a child needs at least four samples at margin $0$ to reach $H = 1$.

**Sparsity-aware splits.** Missing values go left or right, whichever direction gives the higher gain. That default direction is learned.

**Histogram split finding.** Bin each feature into a fixed number of buckets. Fill the per-bucket sums of $G$ and $H$ in one pass over the node, then scan the buckets. The scan costs one step per bucket, not per sorted sample.

## Signal vs. noise

- **Signal:** the tree sees only $g$ and $h$. The leaf weight and gain are two lines of algebra.
- **Signal:** $\lambda$ shrinks continuously, while $\gamma$ prunes discretely.
- **Noise for now:** the weighted quantile sketch, cache layout and sharding. They make it fast, not different.

## Where it breaks

- **Zero hessian with $\lambda = 0$.** The weight is undefined, so keep $\lambda > 0$ or enforce a minimum hessian sum.
- **Saturated samples.** Once $\sigma(\hat m)$ is near 0 or 1, $h \approx 0$. Correctly classified saturated samples also have $g\approx0$ and contribute little. Confidently wrong samples have $|g|\approx1$ and can strongly affect weights and split gains despite their small Hessians.
- **Large steps.** The quadratic model is accurate only near $\hat m$, as the $\delta = 2$ row shows. Shrinkage $\eta$ keeps each step small.
- **Over-regularisation.** A large $\lambda$ pulls every leaf toward zero and underfits small leaves.

## Interview angle

*60-second answer:* Each round, XGBoost fits a tree to the loss's gradient and Hessian. A leaf's best weight is $-G/(H+\lambda)$, and a split's gain is the drop in that regularised objective minus $\gamma$. For tabular data, boosted trees are a strong default. They model thresholds, interactions and missing values directly. A neural network earns its cost when representation learning matters, such as text or images.

*Likely follow-ups:* What do $\lambda$ and $\gamma$ do differently? ($\lambda$ shrinks sizes continuously; $\gamma$ is a discrete threshold on the gain.) Why second order? (It gives the Newton step and one algorithm for any loss.)

## Connections

- **Prerequisites:** `cml/adaboost-and-gradient-boosting` (stagewise fitting), `cml/taylor-expansion-and-newtons-method` (the quadratic model), `cml/choosing-a-tree-split` (the impurity view the gain replaces).
- **Unlocks:** the tree-ensemble topics that follow.
- **Lineage:** Friedman 2001, then Chen & Guestrin 2016. For the interview framing: [library:AI-ML/05-Classification/XGBoost Interview Bridge.md].
- **References:** Chen & Guestrin (2016, §2.2, Eqs. 5–7) give the regularised leaf optimum, the structure score and the split gain [corpus:library-refresh/02-AI-ML/02-Statistical-Learning/2016 -- Chen and Guestrin -- XGBoost - A Scalable Tree Boosting System -- arXiv.pdf.md p.3].
