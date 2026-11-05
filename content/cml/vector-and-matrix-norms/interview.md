# Interview notes — Vector and matrix norms

## Q1. What is the difference between L1 and L2 regularisation, and why does L1 give sparse models?

**60-second answer.** Both add a penalty on the coefficients, scaled by $\lambda$. Ridge penalises
$\lVert\mathbf{w}\rVert_2^2$. It shrinks every coefficient smoothly and has a closed-form solution in
linear regression. The lasso penalises $\lVert\mathbf{w}\rVert_1$. Its unit ball is a diamond with
corners on the axes. The loss contour often first touches the penalty at a corner, where some
coefficients are exactly zero. That is the sparsity. The geometry is the reason, not a
coincidence of the optimiser.

**Follow-ups.**
- *What happens with correlated features?* The lasso tends to pick one feature from a correlated
  group, somewhat arbitrarily. Elastic net mixes both penalties and keeps groups together.
- *Do the penalties need scaled features?* Yes. A penalty on coefficients depends on the units of
  the features, so I standardise first.

## Q2. How do you compute a vector norm safely when the entries are huge or tiny?

**60-second answer.** Squaring the entries overflows once they pass about $10^{154}$, and squaring
tiny entries underflows to zero. I divide by the largest absolute entry first. Every ratio then lies
in $[0, 1]$, so no power overflows, and the sum is at least 1, so it cannot underflow. Then I multiply
the scale back in. This is exact, because the norm is homogeneous. The same idea, subtracting the
maximum before exponentiating, is what makes log-sum-exp and softmax stable.

**Follow-ups.**
- *Does NumPy do this for you?* Not in the version this repo pins. `np.linalg.norm` returns `inf` for
  $[3\cdot10^{200}, 4\cdot10^{200}]$. I would not rely on it for extreme values.
- *What about the zero vector?* Its norm is 0 for every $p$, but dividing by the maximum would divide
  by zero. Check for it first.

## Q3. Where do norms show up in deep-learning training?

**60-second answer.** Two places. Gradient clipping by global norm rescales all the gradients together
when their combined L2 norm exceeds a threshold. That keeps the direction and bounds the step, so one
exploding batch cannot wreck the weights. Spectral normalisation divides a weight matrix by its
largest singular value, which bounds how much the layer can stretch its input, and so bounds its
Lipschitz constant. Weight decay plays the same role as the ridge penalty, though with some optimisers the two are not identical.

**Follow-ups.**
- *Why clip the global norm instead of each gradient?* Clipping each tensor separately changes the
  direction of the update. Clipping the global norm changes only its length.
- *Why is the Frobenius norm sometimes used instead of the spectral norm?* It is cheap, since it needs
  no decomposition. It is an upper bound on the spectral norm, so it is a looser but safe bound.

## Common mistakes

- Calling $\lVert\cdot\rVert_p$ with $p < 1$ a norm, and then relying on the triangle inequality.
- Squaring first and then taking a root, on data with large or tiny entries.
- Confusing the Frobenius norm with the spectral norm when a matrix has many small entries.
- Penalising unscaled coefficients and reading the result as a ranking of feature importance.
