# Interview notes — Logistic regression

## Q1. Derive the gradient of the logistic loss and explain why it is convex.

**60-second answer.** For one sample the Bernoulli log-loss is $\log(1 + e^{\eta}) - y\eta$, with $\eta = b + \mathbf{x}^\top\mathbf{w}$. Its derivative with respect to $\eta$ is $p - y$, where $p = \sigma(\eta)$. The chain rule then gives the gradient $X^\top(\mathbf{p} - \mathbf{y})/n$. The Hessian is $X^\top WX/n$ with $W = \operatorname{diag}(p(1-p))$. Because $W$ has non-negative entries, the Hessian is positive semidefinite, so the loss is convex. An L2 penalty adds $\lambda I$ to the weight block and makes the Hessian positive definite.

**Follow-ups.**
- *Why is the intercept special?* Its gradient is $\frac1n\sum_i(p_i - y_i)$, which is zero at the optimum only if it is not penalised.
- *What does Newton's method do here?* It is iteratively reweighted least squares. Each step solves a weighted least-squares problem with weights $p(1-p)$.

## Q2. Your logistic regression has enormous coefficients. What is happening, and what do you do?

**60-second answer.** The most likely cause is separation: some linear rule splits the classes perfectly. The likelihood then keeps improving as the coefficients grow, so there is no finite maximum, and the solver stops at an arbitrary large value. With both classes present, an L2 weight penalty gives a unique finite minimiser. For one-class data, an unpenalised intercept still diverges. Check first whether a feature leaks the label, or whether the classes are separable in the training data but not in the population.

**Follow-ups.**
- *Does the penalty change the intercept?* Not if it is unpenalised. Penalising the intercept shrinks the base rate towards 0.5 and breaks calibration.
- *How do you pick $\lambda$?* By cross-validated log-loss, on standardised features.

## Common mistakes

- Penalising the intercept, which breaks calibration in the large.
- Summing the loss over samples while the penalty is averaged, so the balance between them depends on $n$.
- Trusting a non-converged fit. Check the gradient at the returned point, not just the iteration count.
- Writing $\log(\sigma(\eta))$ directly. It underflows to $\log 0$ for large negative $\eta$.
