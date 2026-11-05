---
title: Ridge regression and weight decay
minutes: 12
min_viable: "Read The system and Worked example, then solve p01."
---

## The system

Ridge minimises over a scalar intercept $b_0$ and $\boldsymbol\beta\in\mathbb{R}^d$. Here $X\in\mathbb R^{n\times d}$, $\mathbf y\in\mathbb R^n$, $\mathbf 1\in\mathbb R^n$ is all ones, and $I$ is the $d\times d$ identity:

$$\min_{b_0,\boldsymbol\beta}\ \lVert\mathbf y - b_0\mathbf 1 - X\boldsymbol\beta\rVert^2 + \alpha\lVert\boldsymbol\beta\rVert^2,
\qquad \alpha\ge 0.$$

The penalty is on $\boldsymbol\beta$ only, so the intercept is **not** penalised. The objective has no $1/n$ factor,
which matches scikit-learn's `Ridge`. Other texts use $\alpha/2$ or $\alpha/n$, so check which before comparing values of $\alpha$.

Centre $X$ and $\mathbf y$ to remove the intercept: $\tilde X = X - \bar x$ and $\tilde{\mathbf y} = \mathbf y - \bar y$. Then

$$\boldsymbol\beta = (\tilde X^\top\tilde X + \alpha I)^{-1}\tilde X^\top\tilde{\mathbf y},\qquad b_0 = \bar y - \bar x^\top\boldsymbol\beta.$$

| $\alpha$ | Coefficients | Intercept |
|---|---|---|
| $0$ | ordinary least squares (needs full rank) | $\bar y - \bar x^\top\boldsymbol\beta_\text{OLS}$ |
| small, $>0$ | close to OLS, still defined if $\tilde X$ is rank deficient | close to OLS |
| $\to\infty$ | $\to\mathbf 0$ | $\to\bar y$ |

The thin SVD $\tilde X = UDV^\top$ of rank $r$ has positive singular values $d_i$, with columns $\mathbf u_i$ and $\mathbf v_i$:

$$\boldsymbol\beta_\text{ridge} = \sum_i \frac{d_i}{d_i^2+\alpha}\,(\mathbf u_i^\top\tilde{\mathbf y})\,\mathbf v_i.$$

Each direction is shrunk by $d_i^2/(d_i^2+\alpha)$ relative to OLS. Directions with little variance in $X$ (small $d_i$)
shrink the most.

## Worked example

Take $X = (1,2,3)^\top$ and $\mathbf y = (1,2,2)^\top$. Then $\bar x = 2$, $\bar y = 5/3$, $\tilde X = (-1,0,1)^\top$ and
$\tilde{\mathbf y} = (-\tfrac23,\tfrac13,\tfrac13)^\top$. So $\tilde X^\top\tilde X = 2$ and $\tilde X^\top\tilde{\mathbf y} = 1$.

- $\alpha = 0$: $\beta = 1/2$ and $b_0 = 5/3 - 1 = 2/3$.
- $\alpha = 2$: $\beta = 1/(2+2) = 1/4$ and $b_0 = 5/3 - 2\cdot\tfrac14 = 7/6$.

The slope shrinks by the factor $2/(2+2) = \tfrac12$, and the intercept moves toward $\bar y$.

**Feature scale matters.** Take orthogonal centred columns $x_1=(-1,-1,1,1)$ and $x_2=(-1,1,-1,1)$, with $\mathbf y=x_1+x_2$ (squared norms $4$). At $\alpha=10$ each coefficient shrinks from $1$ to $2/7$. Replace $x_2$ by $1000x_2$: its ridge shrink factor becomes $\approx0.9999975$, so $\alpha$ barely penalises it. Standardising restores equal scales.

## Why it works

The penalty adds $\alpha$ to every eigenvalue of $\tilde X^\top\tilde X$. The Hessian $2(\tilde X^\top\tilde X+\alpha I)\succeq 2\alpha I$ is then
positive definite, so the objective is strongly convex and its minimiser is unique even for a rank-deficient design.
Convexity alone would not give uniqueness: a strictly convex objective has at most one optimal point (Boyd and Vandenberghe 2004, §4.2.1, p.137; [source: library-refresh/01-Mathematics/08-Optimization/2004 -- Boyd and Vandenberghe -- Convexity, Duality and KKT Certificates -- Study Note.md]). At $\alpha=0$ a dependent column leaves the coefficients undetermined.

**Weight decay and L2 coincide for plain SGD.** For a loss $L(\mathbf w) + \tfrac{\lambda}{2}\lVert\mathbf w\rVert^2$, one step
of size $\eta$ is $\mathbf w \leftarrow \mathbf w - \eta(\nabla L + \lambda\mathbf w) = (1-\eta\lambda)\mathbf w - \eta\nabla L$.
Shrinking the weights by $(1-\eta\lambda)$ is the same step. Check: with $\mathbf w = (0.8,-1.2)$, $\nabla L = (0.3,-0.1)$,
$\eta = 0.1$ and $\lambda = 0.05$, both forms give $(0.766, -1.184)$.

**Why this breaks for Adam.** Adam scales each coordinate's gradient by a running estimate of its size. L2 adds $\lambda\mathbf w$
to the gradient before that scaling, so the decay is divided by the scale too. At the first bias-corrected Adam step, with zero initial moments, zero data gradient, and negligible $\epsilon$, take $\eta = 0.01$
and $\lambda = 0.1$. A weight of $0.001$ becomes $-0.009$ under Adam with L2, so it overshoots past zero. A weight of $10$
moves by $0.01$ under both rules, because $\eta\lambda\cdot 10=\eta$ here; the small weight shows the difference. Decoupled decay, as in AdamW, multiplies every weight by $(1-\eta\lambda)$ and gives $0.000999$ and $9.99$
(library: AI-ML/03-Optimization/Loshchilov AdamW 2019.md; paper: Loshchilov & Hutter 2017, Decoupled Weight Decay
Regularization, arXiv:1711.05101).

**Choosing $\alpha$.** Score each candidate on held-out folds of the training data, keep the lowest mean error, then refit on all training data, standardising inside each fold.

## Signal vs. noise

- **Signal:** the centred closed form, the unpenalised intercept, and the shrink factor $d_i^2/(d_i^2+\alpha)$.
- **Signal:** standardise features before penalising, and name the $\alpha$ convention you use.
- **Signal:** ridge keeps every feature with a nonzero weight and trades bias for variance. It does not select features.

## Where it breaks

- **Penalising the intercept.** Shifting $\mathbf y$ by a constant then changes the slopes.
- **Unscaled features.** One feature in larger units is penalised less, so $\alpha$ means something different for each feature.
- **Too large $\alpha$.** Every slope goes to zero and the model predicts $\bar y$ everywhere.
- **Dependent columns with $\alpha=0$.** No unique solution exists. Use $\alpha>0$, or drop columns.
- **Adaptive optimisers.** Adam with an L2 term is not weight decay. Use AdamW when you want decoupled decay.

## Interview angle

*"What does ridge do, and how do you choose the penalty?"* Ridge adds $\alpha\lVert\boldsymbol\beta\rVert^2$ to least squares. With
centred data it has a closed form, and it shrinks the directions with little variance most. The penalty is only fair across features
after standardisation. I choose $\alpha$ by K-fold cross-validation on training data, then refit. For plain SGD, weight decay is
L2. For Adam, use decoupled decay (AdamW).

Follow-ups:

- *Why is ridge defined for a rank-deficient design?* Adding $\alpha I$ makes the Gram matrix positive definite.
- *Why does the intercept go to $\bar y$?* With every slope at zero, the best constant is the mean.

Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/normal-equation` (the unpenalised solution), `cml/eigenvalues-and-eigenvectors` (the SVD view),
  `cml/gradient-descent` (the weight-decay step).
- **Unlocks:** `cml/lasso-and-elastic-net` (a different penalty, with exact zeros), `cml/k-fold-cross-validation`,
  `cml/hyperparameter-search`, `cml/bias-variance-decomposition`.
- **Lineage:** Hoerl & Kennard 1970, Ridge regression: biased estimation for nonorthogonal problems (Technometrics). The [AdamW Library note](../../../sources/library/AI-ML/03-Optimization/Loshchilov%20AdamW%202019.md) traces the optimizer connection.
