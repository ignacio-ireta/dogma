---
title: Lagrange multipliers
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Minimise $f(\mathbf{x})$, with $\mathbf{x}\in\mathbb{R}^n$, subject to equality constraints. For linear
constraints $A\mathbf{x} = \mathbf{b}$, where $A\in\mathbb{R}^{m\times n}$ and $\mathbf{b}\in\mathbb{R}^m$, the
objects are:

| Object | Definition | Shape | Role |
|---|---|---|---|
| Lagrangian | $\mathcal{L}(\mathbf{x},\boldsymbol\lambda) = f(\mathbf{x}) - \boldsymbol\lambda^\top(A\mathbf{x}-\mathbf{b})$ | scalar | Folds the constraints into the objective |
| Multiplier | $\boldsymbol\lambda\in\mathbb{R}^m$ | $(m)$ | One per constraint row |
| Stationarity | $\nabla_{\mathbf{x}}\mathcal{L}=0 \iff \nabla f(\mathbf{x}) = A^\top\boldsymbol\lambda$ | $(n)$ | The gradient lies in the row space of $A$ |
| Feasibility | $\nabla_{\boldsymbol\lambda}\mathcal{L}=0 \iff A\mathbf{x}=\mathbf{b}$ | $(m)$ | The constraints hold |

Stationarity and feasibility together are the KKT conditions. For the quadratic
$f = \tfrac12\mathbf{x}^\top Q\mathbf{x} + \mathbf{c}^\top\mathbf{x}$ with $Q$ symmetric, they form one linear system:

$$\begin{pmatrix} Q & -A^\top \\ A & 0 \end{pmatrix}\begin{pmatrix}\mathbf{x}\\ \boldsymbol\lambda\end{pmatrix} = \begin{pmatrix}-\mathbf{c}\\ \mathbf{b}\end{pmatrix}.$$

**Sign convention.** This course puts the minus sign in the Lagrangian, so $\boldsymbol\lambda$ is the
derivative of the optimal value with respect to $\mathbf{b}$. Boyd & Vandenberghe write
$+\nu^\top(A\mathbf{x}-\mathbf{b})$, so their $\nu = -\boldsymbol\lambda$. Check which convention a source
uses before you copy a sign.

## Worked example

Minimise $f = x_1^2 + x_2^2$ subject to $x_1 + x_2 = b$. Here $Q = 2I$, $\mathbf{c} = \mathbf{0}$ and $A = (1\ \ 1)$.

1. Stationarity: $\nabla f = (2x_1, 2x_2) = \lambda(1, 1)$, so $x_1 = x_2 = \lambda/2$.
2. Feasibility at $b = 1$: $\lambda/2 + \lambda/2 = 1$, so $\lambda = 1$ and $\mathbf{x} = (\tfrac12, \tfrac12)$.
3. Optimal value: $p^*(1) = \tfrac14 + \tfrac14 = 0.5$. In general $p^*(b) = b^2/2$.
4. Sensitivity: $p^*(1.01) = 0.51005$, so the forward difference is
   $(0.51005 - 0.5)/0.01 = 1.005$. The true slope is $p^{*\prime}(1) = 1 = \lambda$. The difference
   converges to the multiplier as the step shrinks.

In NumPy, one `np.linalg.solve` on the block matrix returns $\mathbf{x}$ and $\boldsymbol\lambda$ together.

## Why it works

At a constrained minimum the gradient has no component along a feasible direction. Feasible
directions form the null space of $A$, so $\nabla f$ must lie in its orthogonal complement, the row
space of $A$. That is $\nabla f = A^\top\boldsymbol\lambda$.

The sensitivity identity takes one line. Let $\mathbf{x}(\mathbf{b})$ be the minimiser and
$p^*(\mathbf{b}) = f(\mathbf{x}(\mathbf{b}))$. Then

$$dp^* = \nabla f^\top d\mathbf{x} = (A^\top\boldsymbol\lambda)^\top d\mathbf{x} = \boldsymbol\lambda^\top(A\,d\mathbf{x}) = \boldsymbol\lambda^\top d\mathbf{b},$$

because feasibility gives $A\,d\mathbf{x} = d\mathbf{b}$. So $\partial p^*/\partial\mathbf{b} = \boldsymbol\lambda$, and the multiplier is
a shadow price (Boyd & Vandenberghe, *Convex Optimization*, ch. 5).

The KKT matrix is nonsingular when $Q$ is positive definite and $A$ has full row rank. If
$(\mathbf{u}, \mathbf{v})$ solves the homogeneous system, then $A\mathbf{u} = 0$ and $Q\mathbf{u} = A^\top\mathbf{v}$.
Multiplying by $\mathbf{u}^\top$ gives $\mathbf{u}^\top Q\mathbf{u} = (A\mathbf{u})^\top\mathbf{v} = 0$, so $\mathbf{u} = 0$.
Then $A^\top\mathbf{v} = 0$, and full row rank gives $\mathbf{v} = 0$.

## Signal vs. noise

- **Signal:** the KKT system, the sign convention, and $\boldsymbol\lambda = \partial p^*/\partial\mathbf{b}$.
  Interviews expect all three.
- **Signal:** a stationary point is not automatically a minimum (see *Where it breaks*).
- **Noise for now:** constraint qualifications beyond full row rank, the dual function, and duality
  gaps. Inequality constraints add complementary slackness, which is the next step.

## Where it breaks

- **Stationary is not optimal.** Minimise $xy$ on the circle $x^2 + y^2 = 2$. The four KKT points
  are $(1,1)$ and $(-1,-1)$ with $\lambda = \tfrac12$, and $(1,-1)$ and $(-1,1)$ with $\lambda = -\tfrac12$.
  The first pair maximises $xy$ (value $1$) and the second pair minimises it (value $-1$). Here the sign
  of $\lambda$ happens to match the outcome, but a sign is not a test. The second-order condition decides:
  the Hessian of the Lagrangian, restricted to the tangent direction $(1,-1)$ at $(1,1)$, is negative there
  (a maximum) and positive at the minima. Linear constraints plus a convex objective make a KKT point
  the global minimum. Otherwise compare candidates or run the second-order check on the null space of $A$.
- **Dependent constraints.** If the rows of $A$ are dependent, the KKT matrix is singular and
  $\boldsymbol\lambda$ is not unique. A plain solve on nearly dependent rows returns huge numbers
  rather than failing, so check the rank first.
- **Indefinite $Q$.** The stationary point can be a saddle. The contract in p01 requires $Q$
  positive definite for this reason.
- **Sign mix-ups.** A multiplier copied from a source with the other convention has the wrong sign.
  A finite difference of the optimal value catches this.

## Interview angle

Form the Lagrangian with one multiplier per constraint row. Stationarity and feasibility give the
KKT system, which is one linear solve of size $n+m$. Say that the matrix is indefinite, so LU is
the right tool, not Cholesky. The multiplier is the shadow price of the constraint. Without
convexity, a KKT point is only a candidate. Follow-ups: what $\lambda$ means (the derivative of the
optimal value in $\mathbf{b}$), and why a KKT point can be a maximum (the circle example). See
`interview.md` for the full answers.

## Connections

- **Prerequisites:** `cml/partial-derivatives-and-gradients` (stationarity), `cml/hessians-definiteness-and-critical-points`
  (positive definiteness and second-order tests), `cml/matrix-inverse-and-rank` (full row rank).
- **Unlocks:** `cml/support-vector-machines`, whose margin constraints are inequalities. Their
  multipliers are nonzero only at the support vectors.
- **Lineage:** Lagrange's *Mécanique analytique* (1788) used multipliers for constrained mechanics.
  Karush (1939) and Kuhn & Tucker (1951) extended the conditions to inequality constraints.
