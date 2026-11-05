# Interview notes: Gradient descent

## Q1. Walk me through gradient descent, and what goes wrong when the learning rate is too large.

**60-second answer.** Gradient descent repeats $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\nabla f(\mathbf{x}_k)$.
On a quadratic with curvature matrix $A$, each eigen-direction of the error shrinks by
$|1 - \eta\lambda_i|$ per step, so the method converges exactly when $0 < \eta < 2/\lambda_{\max}$.
Past that limit the largest-curvature direction overshoots and grows. Just past $1/\lambda_{\max}$
it oscillates in sign while still shrinking. Beyond $2/\lambda_{\max}$ it diverges, and iterates
overflow to inf or NaN. In practice I check for non-finite iterates and stop with a clear error.
For a general smooth function the quadratic bound is only local. The global guarantee needs
convexity: with a convex $L$-smooth $f$ and $\eta\le 1/L$, the objective gap falls as $O(1/k)$.

**Follow-ups.**
- *Why $2/\lambda_{\max}$?* On a quadratic the update is linear, so stability is set by the factor
  $|1 - \eta\lambda|$ in the eigen-direction with the largest $\lambda$.
- *What if the curvature is ill-conditioned?* The step is capped by $\lambda_{\max}$, but progress
  along the flat direction is set by $\lambda_{\min}$. The steps needed grow roughly with
  $\kappa = \lambda_{\max}/\lambda_{\min}$. Scaling the features or preconditioning reduces $\kappa$.
- *Does the $2/L$ limit carry over to nonconvex losses?* Not as a guarantee of a global minimum. For
  $L$-smooth nonconvex $f$ and $\eta<2/L$ the objective decreases and the gradient norms go to zero,
  but the limit can be a saddle or a maximum.

## Q2. How would you decide when to stop?

**60-second answer.** Use two rules together: stop when the gradient norm is below a tolerance,
and stop after a maximum number of steps. The gradient rule is about optimality, but it depends on
the scale of the objective, so I would also watch the change in the loss or compare with the initial
gradient norm. The step budget guards against a learning rate that is too small. For a
production training loop, the stopping rule is usually validation loss rather than the gradient.

**Follow-ups.**
- *Why not stop when the step is small?* A small step can come from a small learning rate, not from
  being near the minimum. The gradient norm does not have that problem.
- *How do you count steps?* Check the rule before the update, so that $k$ means $k$ updates were
  applied.

## Q3. Why does gradient descent zig-zag on some problems?

**60-second answer.** When the level sets are long, thin ellipses, the gradient points across the
narrow direction, not toward the minimum. Each step overshoots in the steep direction and makes
little progress along the valley floor. The path alternates sides of the valley and converges slowly.
Momentum, a per-coordinate scaling such as Adam, or a second-order method reduces this, and
conjugate gradients solve a quadratic in at most $d$ steps in exact arithmetic.

**Follow-ups.**
- *Is zig-zag always a bug?* No. It is a symptom of conditioning, and the fix is to change the
  problem or the method, not to tune the stopping rule.
- *Which is cheaper, a fixed step or a line search?* A fixed step costs one gradient per step. A
  line search costs extra function evaluations per step but needs no curvature estimate.

## Common mistakes

- Stepping uphill: the update must subtract the gradient.
- Counting steps after the update, which reports one extra step.
- Checking the gradient norm against a tolerance that depends on the scale of the objective.
- Modifying the starting point in place, which breaks any caller that reuses it.
- Accepting a non-positive learning rate, or not checking for overflow, so divergence returns NaN
  instead of an error.
