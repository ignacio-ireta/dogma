# From Derivatives to Jacobians and Hessians

**Scope:** modern mathematical synthesis, using introductory textbooks and proof-based analysis. The 2026 date identifies this library synthesis, not the invention of these ideas.

A tangent slope answers how one number responds to another. Learning problems require the response of many outputs to many parameters, and an optimizer must decide how far a local approximation can be trusted. The bridge is to treat the derivative as a linear map, then measure how that map changes.

For an open set $U\subset\mathbb R^n$, differentiability of $f:U\to\mathbb R^m$ at $x$ means that a linear map $Df(x)$ satisfies
$$
f(x+h)=f(x)+Df(x)h+r(h),\qquad
\frac{\|r(h)\|}{\|h\|}\longrightarrow0.
$$
In coordinates, this map is the $m\times n$ Jacobian $J_f(x)$. Partial derivatives describe coordinate directions; differentiability requires one approximation valid over **all** small directions. Existence of all partial derivatives alone does not establish that property. Continuity of the partial derivatives in a neighborhood is a useful sufficient condition.

For example, define $f(0,0)=0$ and
$$
f(x,y)=\frac{xy}{\sqrt{x^2+y^2}}\quad ((x,y)\ne(0,0)).
$$
Both partial derivatives at the origin vanish. Along $(t,t)$, however, $f(t,t)=|t|/\sqrt2$, and the error divided by $\|(t,t)\|$ is $1/2$. The zero linear map does not approximate the function as required. This is why merely assembling a list of slopes can miss the geometry.

The chain rule becomes matrix composition:
$$
J_{g\circ f}(x)=J_g(f(x))J_f(x).
$$
For a scalar objective, $Df(x)h=\nabla f(x)^\top h$. Forward differentiation propagates $Jh$; reverse differentiation propagates $J^\top v$. Neither operation requires storing the full Jacobian. [Automatic differentiation](2018%20--%20Baydin%20et%20al%20--%20Automatic%20Differentiation%20as%20Executable%20Chain%20Rules%20--%20Study%20Note.md) makes this distinction computational.

If $f$ is $C^2$ near $x$, the Hessian is symmetric and gives the quadratic approximation
$$
f(x+h)=f(x)+\nabla f(x)^\top h+
\tfrac12h^\top\nabla^2f(x)h+o(\|h\|^2).
$$
The scalar Taylor theorem applied to $t\mapsto f(x+th)$ explains the formula. At a stationary point, a positive definite Hessian gives a strict local minimum; an indefinite Hessian exposes directions of opposite curvature. A semidefinite Hessian alone may leave the question undecided.

Consider least squares:
$$
f(w)=\tfrac12\|Aw-b\|^2,\quad
\nabla f=A^\top(Aw-b),\quad H=A^\top A.
$$
Here the approximation is exact. Null directions of $A$ have zero curvature, revealing nonuniqueness when they survive the constraints. Adding $\lambda\|w\|^2/2$ with $\lambda>0$ makes $H+\lambda I$ positive definite. When $H$ is invertible, Newton's step solves $Hp=-\nabla f$; its reliability outside a quadratic neighborhood requires step control.

Invertibility has a related local meaning. A $C^1$ map with invertible derivative at a point has a differentiable inverse on suitable neighborhoods. This does not establish global invertibility. The implicit function theorem similarly converts a regular equation into a local function; [optimization layers](../08-Optimization/2022%20--%20Blondel%20et%20al%20--%20Implicit%20Differentiation%20and%20Optimization%20Layers%20--%20Study%20Note.md) use precisely that conversion.

The conceptual route is compositional differentiation → linear approximation → Jacobian products → curvature and constrained sensitivity. This is a prerequisite map, not a claim of direct historical influence. It explains why neural-network training, inverse problems, and sensitivity analysis share the same calculus.

## Sources and locators

- Herman and Strang, Calculus Volume 1, [official §3.1](https://openstax.org/books/calculus-volume-1/pages/3-1-defining-the-derivative): derivative as a limit. Their Volume 3, [official §4.4](https://openstax.org/books/calculus-volume-3/pages/4-4-tangent-planes-and-linear-approximations), treats total differentiability and linear approximation. These are introductory secondary expositions.
- Lebl, Basic Analysis I, version 6.3, §4.3, printed pp.171–175; Basic Analysis II, version 6.3, §§8.3–8.6, printed pp.35–64; inverse theorem 8.5.1, p.51, implicit theorem 8.5.6, p.55. [Author source](https://www.jirka.org/ra/).
- Boyd and Vandenberghe, Convex Optimization, appendix A.4, pp.640–644, and §9.5, pp.484–495. [Author-approved book](https://web.stanford.edu/~boyd/cvxbook/). Printed page numbers are used throughout.

