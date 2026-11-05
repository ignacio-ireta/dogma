# Interview notes: Lagrange multipliers

## Q1. How do you minimise a quadratic subject to linear equality constraints?

**60-second answer.** Form the Lagrangian with one multiplier per constraint row. Stationarity gives
$Q\mathbf{x} + \mathbf{c} = A^\top\boldsymbol\lambda$, and feasibility gives $A\mathbf{x} = \mathbf{b}$. Stack them into
the KKT system, one linear solve of size $n+m$. The matrix is indefinite, so use a general solver such
as LU, not Cholesky. Check two things first: $Q$ is positive definite, and $A$ has full row rank. The
first makes the minimum unique, and the second makes the system solvable with unique multipliers.
Do not form an explicit inverse.

**Follow-ups.**
- *What does $\boldsymbol\lambda$ mean?* It is the derivative of the optimal value with respect to the
  constraint level: $\partial p^*/\partial\mathbf{b} = \boldsymbol\lambda$.
- *When is the KKT point the global minimum?* When the objective is convex and the constraints are
  linear. Then every KKT point is a global minimum.

## Q2. What does a Lagrange multiplier tell you?

**60-second answer.** It is the shadow price of the constraint. It gives the marginal change in the
optimal objective per unit change in the constraint level. A large magnitude means the constraint
binds hard. The sign depends on the Lagrangian convention, so state the convention. The
derivation is one line: $dp^* = \boldsymbol\lambda^\top d\mathbf{b}$, from stationarity and $A\,d\mathbf{x} = d\mathbf{b}$.

**Follow-ups.**
- *How do you check the sign?* Take a finite difference of the optimal value in $\mathbf{b}$.
- *Where does this show up in ML?* In SVM margin constraints. Only the support vectors have nonzero
  multipliers.

## Q3. Is a KKT point the answer?

**60-second answer.** Not by itself. A KKT point is a stationary, feasible candidate. Without
convexity it can be a maximum or a saddle. Minimise $xy$ on $x^2 + y^2 = 2$: the four KKT points
include two minima and two maxima. The sign of $\lambda$ is not a test: here $\lambda = +\tfrac12$ marks
the maxima, but a second-order check on the tangent direction is what separates them. For a convex
objective with linear constraints, a KKT point is sufficient. Otherwise compare the candidates or
check the second-order condition on the null space of $A$.

**Follow-ups.**
- *What is the second-order check?* The Hessian of the Lagrangian must be positive definite on the
  null space of $A$ for a local minimum.
- *Why does this matter for training?* Non-convex losses have stationary points that are not
  minima, so a zero gradient is not evidence of a solution.

## Common mistakes

- Mixing the two sign conventions for $\boldsymbol\lambda$, which flips the reported sensitivity.
- Dropping $\mathbf{c}$ from the right-hand side, or solving for $\mathbf{x}$ and forgetting the constraints.
- Calling a KKT point the minimum without a convexity or second-order argument.
- Using Cholesky on the KKT matrix. It is indefinite, so the factorisation does not exist.
- Forming the inverse of the KKT matrix instead of solving the system.
- Skipping the rank check. Dependent constraints give a singular system or non-unique multipliers.
