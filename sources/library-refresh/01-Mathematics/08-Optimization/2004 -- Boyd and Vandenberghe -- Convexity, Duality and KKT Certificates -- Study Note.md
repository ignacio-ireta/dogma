# Convexity, Duality and KKT Certificates

**Work:** Stephen Boyd and Lieven Vandenberghe, *Convex Optimization* (2004). This is a study of their modern textbook treatment, not an attribution of convex duality or KKT to the textbook's authors.

Finding a stationary point answers only a local question. Estimation also imposes restrictions: coefficients must remain nonnegative, probabilities must sum to one, or a resource budget must hold. Convexity supplies global structure; duality supplies lower bounds; optimality conditions connect a candidate solution to a certificate.

A function on a convex domain is convex when
$$
f((1-t)x+ty)\le(1-t)f(x)+tf(y),\quad0\le t\le1.
$$
For differentiable $f$, this is equivalent to the supporting-plane inequality $f(y)\ge f(x)+\nabla f(x)^\top(y-x)$. Thus an unconstrained zero gradient certifies a global minimum. Convexity does not itself ensure a minimum exists, nor that it is unique; strict or strong convexity addresses uniqueness when a minimum is attained.

Consider
$$
\min_x f_0(x)\quad\text{subject to }f_i(x)\le0,\quad Ax=b,
$$
with convex $f_0,f_i$. The Lagrangian is
$$
L(x,\lambda,\nu)=f_0(x)+\sum_i\lambda_i f_i(x)+\nu^\top(Ax-b),
\qquad\lambda\ge0.
$$
Its dual function $g(\lambda,\nu)=\inf_xL(x,\lambda,\nu)$ is a lower bound on every feasible objective value: the inequality terms are nonpositive at a feasible $x$. Maximizing this bound is the dual problem. Weak duality always gives $d^\star\le p^\star$, including beyond the convex setting.

Strong duality needs additional conditions. A standard sufficient condition for convex problems is Slater's condition: a feasible point in the relative interior of the common domain satisfies all nonaffine inequalities strictly. Appropriate finiteness assumptions also give attainment of the dual optimum. Convexity alone should not be used as a substitute for a constraint qualification.

For differentiable objectives and constraints, the KKT conditions are primal feasibility, $\lambda_i\ge0$, complementary slackness $\lambda_i f_i(x)=0$, and stationarity
$$
\nabla f_0(x)+\sum_i\lambda_i\nabla f_i(x)+A^\top\nu=0.
$$
In a convex problem, any point and multipliers satisfying these conditions certify global optimality. Under Slater's condition they also characterize an attained optimum. In a general nonconvex problem, necessary KKT conditions require regularity and are not a global certificate.

For a concrete check, minimize $\tfrac12(x-a)^2$ subject to $x\ge0$. Writing the constraint as $-x\le0$ gives $x-a-\lambda=0$, $\lambda x=0$. The solution is $x^\star=\max(a,0)$: for $a<0$, the active constraint has multiplier $-a$; for $a>0$, the multiplier is zero. At the boundary, feasibility replaces the unconstrained requirement that the objective gradient vanish.

Multipliers also measure sensitivity. When an optimal value is differentiable with respect to a relaxed bound $f_i(x)\le u_i$, its derivative is $-\lambda_i^\star$ under the relevant regularity assumptions. This turns a certificate into information about which constraint is costly.

The learning connection is concrete: least squares, logistic regression, support-vector estimation, and many regularized inverse problems admit convex formulations. [Proximal methods](2026%20--%20Library%20--%20Proximal%20Steps%20and%20Accelerated%20Convex%20Optimization%20--%20Concept%20Note.md) exploit their structure at scale. [Differentiable optimization layers](2022%20--%20Blondel%20et%20al%20--%20Implicit%20Differentiation%20and%20Optimization%20Layers%20--%20Study%20Note.md) go further: they differentiate the equations certifying an optimum. This is a mathematical reuse of KKT, not a claim that all neural-network training is convex.

## Sources and locators

- Boyd and Vandenberghe, Convex Optimization: §3.1, pp.67–76 (convexity tests); §4.2, pp.136–145 (convex problems); §5.1, pp.215–222 (dual function); §5.2.3, pp.226–227 (Slater); §5.5.3, pp.243–246 (KKT); §5.6, pp.249–253 (sensitivity); chapter 7, pp.351–396 (estimation). [Author-approved complete book](https://web.stanford.edu/~boyd/cvxbook/). All page numbers are printed pages.
- Amos and Kolter, OptNet (ICML 2017), §3, proceedings pp.138–139 (PDF pp.3–4), differentiates QP optimality conditions. [Official PMLR record](https://proceedings.mlr.press/v70/amos17a.html).

