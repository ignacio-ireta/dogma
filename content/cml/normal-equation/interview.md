# Interview notes: Normal equation

## Q1. Derive the least-squares solution, and say what it means geometrically.

**60-second answer.** Minimise $L(\boldsymbol\beta) = \lVert\mathbf{y} - A\boldsymbol\beta\rVert^2$ for a
design $A$ of shape $n\times p$. The gradient is $-2A^\top(\mathbf{y} - A\boldsymbol\beta)$. Setting it
to zero gives the normal equations $A^\top A\boldsymbol\beta = A^\top\mathbf{y}$. Geometrically, the
fitted values $A\hat{\boldsymbol\beta}$ are the orthogonal projection of $\mathbf{y}$ onto the column space
of $A$. The residual is orthogonal to every column, which is exactly the condition $A^\top\mathbf{r} = 0$.
When $A$ has full column rank, $A^\top A$ is invertible and the solution is unique.

**Follow-ups.**
- *Why is the stationary point the minimum?* $L$ is convex, because its Hessian $2A^\top A$ is positive
  semidefinite. A stationary point of a convex function is a global minimum.
- *What is the hat matrix?* $H = A(A^\top A)^{-1}A^\top$, the projection onto the column space. It is
  symmetric and idempotent.

## Q2. When would you not use the normal equations?

**60-second answer.** When the design is ill-conditioned, forming $A^\top A$ squares the condition
number, so $\kappa(A^\top A) = \kappa(A)^2$ in the 2-norm, and precision is lost. In that case I would
use a QR factorisation or an SVD-based `lstsq`, which never forms $A^\top A$. When $p$ is very large, the
cost of forming $A^\top A$, which is $O(np^2)$, may be too high, so I would use an iterative method such as
gradient descent or conjugate gradients. When the design is rank deficient, I would fix the features or
add a ridge penalty, because no unique solution exists.

**Follow-ups.**
- *Why does squaring matter in practice?* A design with $\kappa(A) = 10^4$ gives $\kappa(A^\top A) = 10^8$,
  so the error bound $\kappa(A^\top A)\,\varepsilon$ is about $10^{-8}$: roughly eight correct digits at
  best, and fewer along the nearly flat direction.
- *Is the normal equation ever the right choice?* Yes, for small, well-conditioned problems, and when the
  Gram matrix itself is wanted (for example, for an inference covariance).

## Q3. Collinear features or a one-hot encoding are in the model. What happens?

**60-second answer.** The design is rank deficient, so $A^\top A$ is singular and the coefficients are not
identifiable. A one-hot encoding together with an intercept is the classic dummy-variable trap, because the
dummy columns sum to the intercept column. The fix is to drop one level, which makes the others relative to
it. If all features must stay, a ridge penalty $\lambda I$ makes the system invertible for any $\lambda>0$.
I would also check the rank before solving, because a plain solve can return arbitrary coefficients silently.

**Follow-ups.**
- *Does rank deficiency hurt predictions?* Not always. The fitted values can be stable even when the
  coefficients are not, but the individual coefficients have no meaning.
- *How would you detect near-collinearity?* Look at the singular values of the design, or at the variance
  inflation factors.
- *Does a small residual show the coefficients are right?* Not for nearly collinear designs. In one
  example the residual stayed below $10^{-7}$ while the coefficients were off by $0.2$ or more.

## Common mistakes

- Solving with `inv(AᵀA) @ Aᵀy`, which is slower and less accurate than a solve.
- Forgetting the intercept column, so the fit is forced through the origin.
- Writing the intercept as an array rather than a float, or the coefficients as a column vector.
- Checking the rank of $A^\top A$ with an exact zero test, which misses near-singular designs.
- Dropping a one-hot level without noticing that the intercept already covers it.
