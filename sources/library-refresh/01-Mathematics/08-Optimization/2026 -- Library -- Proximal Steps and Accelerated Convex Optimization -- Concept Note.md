# Proximal Steps and Accelerated Convex Optimization

**Scope:** modern synthesis using Parikh–Boyd and Bubeck. Their texts explain established algorithms; they are not the original discoveries of proximal operators or Nesterov acceleration.

Gradient descent assumes the objective has a usable derivative. Sparse estimation introduces a kink through an $\ell_1$ penalty; constraints introduce a boundary. Replacing these structures by smooth approximations can change the problem. Proximal methods preserve the difficult term and solve a small local optimization problem instead.

Let $F=f+g$, where $f$ is convex with $L$-Lipschitz gradient, and $g$ is proper, closed, convex, and possibly extended-valued. Define
$$
\operatorname{prox}_{t g}(v)=
\arg\min_x\left\{g(x)+\frac{\|x-v\|^2}{2t}\right\},\qquad t>0.
$$
The quadratic term makes this subproblem strongly convex and gives a unique minimizer. If $g$ is the indicator of a nonempty closed convex set, the proximal operator is Euclidean projection. Thus constraints and penalties use the same machinery.

The proximal-gradient update is
$$
x_{k+1}=\operatorname{prox}_{t g}(x_k-t\nabla f(x_k)).
$$
It minimizes the local model consisting of the linearization of $f$, the exact term $g$, and a quadratic penalty for moving. For $t\le1/L$, the model majorizes the smooth contribution. Under the standard convex assumptions, an attained minimum, and suitable initialization, this gives an $O(1/k)$ objective-gap bound.

For the lasso,
$$
F(x)=\tfrac12\|Ax-b\|^2+\lambda\|x\|_1,
$$
a gradient step handles the squared residual. The proximal step acts coordinatewise:
$$
[\operatorname{prox}_{t\lambda\|\cdot\|_1}(v)]_i
=\operatorname{sign}(v_i)\max(|v_i|-t\lambda,0).
$$
This soft thresholding produces exact zeros. It is the solution of a one-dimensional convex problem at each coordinate, not an arbitrary rule for pruning weights. A valid Lipschitz constant is $\|A\|_2^2$.

Acceleration addresses a separate limitation: even smooth convex gradient descent can approach the optimum slowly. A FISTA form starts with $s_1=1$, $y_1=x_0$, and repeats
$$
x_k=\operatorname{prox}_{g/L}\!\left(y_k-\nabla f(y_k)/L\right),\qquad
s_{k+1}=\frac{1+\sqrt{1+4s_k^2}}2,
$$
$$
y_{k+1}=x_k+\frac{s_k-1}{s_{k+1}}(x_k-x_{k-1}).
$$
For the same convex composite setting, this achieves an $O(1/k^2)$ objective-gap rate. Extrapolation is tied to a proof and a particular schedule; adding an arbitrary momentum coefficient does not inherit the theorem. Individual objective values may oscillate.

The hypothesis distinctions matter in AI. Convexity enables global objective-gap guarantees. Smoothness bounds how badly a local linear model can mispredict. Strong convexity gives additional curvature and can improve rates. A nonsmooth regularizer does not destroy the theory when its proximal map remains tractable. None of this automatically transfers to a nonconvex neural loss or to noisy gradients; [stochastic optimization](2018%20--%20Bottou%2C%20Curtis%20and%20Nocedal%20--%20Stochastic%20Gradients%2C%20Noise%20and%20Computation%20--%20Study%20Note.md) has another error term.

The conceptual chain is [quadratic local models](../03-Calculus/2026%20--%20Library%20--%20From%20Derivatives%20to%20Jacobians%20and%20Hessians%20--%20Concept%20Note.md) → [convex certificates](2004%20--%20Boyd%20and%20Vandenberghe%20--%20Convexity%2C%20Duality%20and%20KKT%20Certificates%20--%20Study%20Note.md) → structured local minimization → acceleration. Splitting methods and differentiable solvers reuse these operators. The practical lesson is to inspect the objective's structure before choosing an update rule: inexpensive exact handling of one term can be more useful than differentiating everything.

## Sources and locators

- Parikh and Boyd, Proximal Algorithms, inspected **2013 author manuscript**: §1.1, p.124 (definition); §4.2, pp.148–151 (proximal gradient); §4.3, p.152 (acceleration); §6.5, pp.187–190 (norms); §7.1, pp.196–200 (lasso). [Authors' page](https://web.stanford.edu/~boyd/papers/prox_algs.html). The linked PDF carries 2013 and a placeholder DOI; the published 2014 monograph has different pagination, [publisher metadata](https://www.nowpublishers.com/article/DownloadSummary/OPT-003).
- Bubeck, Convex Optimization - Algorithms and Complexity (2015, arXiv v2), §3.2, pp.266–271; §3.7, pp.289–295; §5.1, pp.310–312. [Versioned author-deposited source](https://arxiv.org/abs/1405.4980v2). These cover gradient bounds, Nesterov acceleration, and FISTA.
- Boyd and Vandenberghe, Convex Optimization, §9.3, pp.466–475, for the smooth-descent baseline. [Complete author-approved book](https://web.stanford.edu/~boyd/cvxbook/). Locators use printed pages.

