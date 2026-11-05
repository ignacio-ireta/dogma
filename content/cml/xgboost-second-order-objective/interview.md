# Interview notes — XGBoost second-order objective

## Q1. Derive the optimal leaf weight and the split gain from the gradient and Hessian.

**60-second answer.** For each sample, the loss gives a gradient $g_i$ and a Hessian $h_i$ at the current margin. A second-order Taylor model of the loss in a leaf is $Gw + \tfrac12(H+\lambda)w^2$, with $G$ and $H$ the sums over the leaf. Minimising it gives $w^* = -G/(H+\lambda)$. The leaf's minimum is $-\tfrac12 G^2/(H+\lambda)$ plus $\gamma$. A split's gain is the sum of the two positive child scores minus the positive parent score, halved, minus $\gamma$. That is the formula $\tfrac12[G_L^2/(H_L+\lambda) + G_R^2/(H_R+\lambda) - (G_L+G_R)^2/(H_L+H_R+\lambda)] - \gamma$. Every candidate split costs one pass over its sums, so the search only needs $g$ and $h$.

**Follow-ups.**
- *Why does the Hessian belong in the denominator?* It is the curvature of the loss. A flat region (small $h$) takes a bigger step, and a steep region takes a smaller one.
- *What if $H = 0$ on one side?* With $\lambda > 0$ the weight is $-G/\lambda$, which is defined. With $\lambda = 0$ it is undefined, so the side must be rejected.

## Q2. What do $\lambda$ and $\gamma$ do, and how do they differ?

**60-second answer.** $\lambda$ is an L2 penalty on the leaf weights. It shrinks every leaf toward zero, the shrinkage is smooth in $\lambda$, and positive $\lambda$ keeps $H+\lambda$ positive. $\gamma$ is a fixed cost per leaf. It enters the gain as a constant, so it decides which splits are kept, not how big each leaf's weight is. A split is kept only when its gain after $\gamma$ is positive. So $\lambda$ controls the size of the fitted values, and $\gamma$ controls the number of leaves.

**Follow-ups.**
- *What does `min_child_weight` do?* It sets a minimum on the Hessian sum in each child. With logistic loss, each sample's $h_i \le 0.25$, so the threshold forbids children with few uncertain samples.
- *How would you tune them?* Use validation log-loss, with $\lambda$ and $\gamma$ searched on a log scale. A large gap between train and validation loss points to increasing them.

## Q3. Why would you use XGBoost rather than a neural network on tabular data?

**60-second answer.** Tabular data is a set of mixed features where the important structure is thresholds, interactions and missing values. Trees model those directly, and boosting builds an additive model of them. XGBoost adds second-order split scoring, regularised leaf weights and sparsity-aware splits, so it is a strong default on modest data. It trains quickly, needs little tuning, and gives feature importances that stakeholders can read. A neural network is the better choice when representation learning matters, such as text, images or very large data. The choice is about the shape of the problem, not which model is more advanced.

**Follow-ups.**
- *Would you ever use a neural network on tabular data?* Yes, when there is enough data and a clear reason, such as learned embeddings for high-cardinality categories. The cost is more tuning and less interpretability.
- *What would you check before trusting the comparison?* The same split, the same features and the same tuning budget for both models, and calibration as well as ranking.

## Common mistakes

- Dropping $\lambda$ from the leaf weight's denominator and writing $-G/H$. With $H$ near zero the weight blows up on small leaves.
- Treating $\gamma$ as a per-sample penalty. It is a per-leaf cost, so it is not scaled by the number of samples.
- Computing the sigmoid as $1/(1+e^{-m})$ for large negative $m$. It overflows and raises warnings. Writing the Hessian as $e^{m}/(1+e^{m})^2$ is worse: for large $m$ it gives $\infty/\infty$, which is NaN. The form $\sigma(1-\sigma)$ stays finite.
- Forgetting that the second-order model is local. Shrinkage keeps each step small, and that is why it matters.
