# Implicit Differentiation and Optimization Layers

**Works:** Blondel et al., *Efficient and Modular Implicit Differentiation* (NeurIPS 2022), with Amos and Kolter's *OptNet* (ICML 2017). These are research applications of classical implicit-function and optimality mathematics.

A layer need not be an explicit formula. It can return the solution of a constrained estimation problem, a fixed point, or an equilibrium. Differentiating every iteration of the algorithm that finds the answer can require substantial memory and can tie the derivative to a chosen stopping procedure. An alternative is to differentiate the equation defining the solution.

Suppose $F(y,\theta)=0$ defines $y^\star$ at a parameter $\theta$. Assume $F$ is continuously differentiable near the point and the square matrix $F_y$ is invertible there. The implicit function theorem gives a locally unique differentiable branch with
$$
\frac{dy^\star}{d\theta}=-F_y^{-1}F_\theta.
$$
The inverse denotes a mathematical operation; implementations should ordinarily solve linear systems. A small singular value in $F_y$ indicates potentially large sensitivity and numerical difficulty, even when invertibility technically holds.

For a scalar outer loss $L(y^\star,\theta)$, solve
$$
F_y^\top v=\nabla_y L.
$$
Then
$$
\frac{dL}{d\theta}=\partial_\theta L-v^\top F_\theta.
$$
Jacobian–vector and transposed-Jacobian products let automatic differentiation evaluate the needed actions without materializing every matrix. This connects [executable chain rules](../03-Calculus/2018%20--%20Baydin%20et%20al%20--%20Automatic%20Differentiation%20as%20Executable%20Chain%20Rules%20--%20Study%20Note.md) to numerical linear algebra.

Ridge regression gives a complete example:
$$
y^\star(\lambda)=\arg\min_y
\tfrac12\|Xy-b\|^2+\tfrac{\lambda}{2}\|y\|^2,\quad \lambda>0.
$$
Its defining equation is $F=(X^\top X+\lambda I)y-X^\top b=0$. Since $H=X^\top X+\lambda I$ is positive definite,
$$
\frac{dy^\star}{d\lambda}=-H^{-1}y^\star.
$$
A validation loss can therefore adjust the regularization strength by an outer gradient. This is a simple bilevel problem: one optimization defines the parameter used by another.

Constrained layers use [KKT equations](2004%20--%20Boyd%20and%20Vandenberghe%20--%20Convexity%2C%20Duality%20and%20KKT%20Certificates%20--%20Study%20Note.md) for primal variables and multipliers. OptNet applies this idea to parameterized quadratic programs. Smooth dependence requires additional regularity of those equations, often involving constraint independence, curvature, and stability of the active constraints. Strong convexity of the objective guarantees a unique primal solution but does not guarantee global differentiability in every parameter.

For example,
$$
y^\star(\theta)=\arg\min_{y\ge0}\tfrac12(y-\theta)^2
=\max(0,\theta)
$$
is unique for every $\theta$, yet has a kink at zero. That is an active-set transition. An implicit derivative valid on one regular branch must not be mistaken for a derivative at the transition.

Blondel et al. make the procedure modular: specify a residual or optimality condition, use autodiff on that condition, and attach an implicit derivative to a solver. The residual equation and the solver implementation can be separated. An approximate root still produces an approximate derivative; residual size, conditioning, smoothness, and linear-solve accuracy all matter. A small residual alone does not ensure a small gradient error in an ill-conditioned system.

The conceptual route is local invertibility → constrained optimality → sensitivity of a solution map → differentiable estimators and learned decision layers. This extends trainable models to components with explicit structure. It also clarifies a modeling choice: differentiating a finite iteration sequence targets that sequence; implicit differentiation targets a regular solution branch. Agreement depends on convergence and numerical accuracy.

## Sources and locators

- Blondel et al., Efficient and Modular Implicit Differentiation, §2.1, pp.2–3 (implicit theorem and Jacobian products); §2.2, pp.4–6 (optimality conditions and examples); §3, pp.6–7 (Jacobian precision guarantees for approximate solutions). [Official NeurIPS publication](https://proceedings.neurips.cc/paper_files/paper/2022/hash/228b9279ecf9bbafe582406850c57115-Abstract-Conference.html). Conference year 2022 is distinguished from the 2021 preprint.
- Amos and Kolter, OptNet, §3, proceedings pp.138–139 (PDF pp.3–4), and §3.1.1, p.139, for QP differentiation and factorization reuse. [Official ICML/PMLR record](https://proceedings.mlr.press/v70/amos17a.html).
- Lebl, Basic Analysis II, version 6.3, theorem 8.5.6, printed pp.55–57. [Author edition](https://www.jirka.org/ra/realanal2.pdf).
- Boyd and Vandenberghe, Convex Optimization, §5.5.3, pp.243–246, and §5.6, pp.249–253, for optimality and sensitivity. [Author-approved book](https://web.stanford.edu/~boyd/cvxbook/). The ridge and projection examples above are direct worked applications.

