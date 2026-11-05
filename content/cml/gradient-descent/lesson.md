---
title: Gradient descent
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

To minimise a differentiable $f:\mathbb{R}^d\to\mathbb{R}$, start at $\mathbf{x}_0\in\mathbb{R}^d$
and repeat

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \eta\,\nabla f(\mathbf{x}_k),$$

where $\eta>0$ is the **learning rate** (step size) and $\nabla f(\mathbf{x}_k)\in\mathbb{R}^d$ is the
gradient, the direction of steepest local increase. Stepping against it is the steepest local
decrease per unit of step length. An implementation makes three decisions:

| Decision | Rule | Failure mode if wrong |
|---|---|---|
| Step length | fixed $\eta$ (ML default) or a line search | too large: divergence; too small: no progress |
| Stop | $\lVert\nabla f(\mathbf{x}_k)\rVert\le\varepsilon$, or a step budget, whichever comes first | stop too early, or never |
| Count | $n$ = number of updates applied | off by one if the stop test runs after the update |

The stop test on the gradient is about optimality, but it depends on the scale of $f$. Multiply $f$
by $1000$ and the same point fails the test.

## Worked example

Take $f(\mathbf{x}) = \tfrac12\mathbf{x}^\top A\mathbf{x} - \mathbf{b}^\top\mathbf{x}$ with
$A = \operatorname{diag}(1, 10)$ and $\mathbf{b} = (1, 10)$. The gradient is $A\mathbf{x} - \mathbf{b}$
and the minimiser is $\mathbf{x}^* = A^{-1}\mathbf{b} = (1, 1)$. Start at $\mathbf{x}_0 = (0, 0)$ with
$\eta = 0.1$.

1. $\nabla f(\mathbf{x}_0) = (0 - 1,\, 0 - 10) = (-1, -10)$, so $\mathbf{x}_1 = (0,0) - 0.1(-1,-10) = (0.1,\ 1.0)$.
2. $\nabla f(\mathbf{x}_1) = (0.1 - 1,\ 10 - 10) = (-0.9,\ 0)$, so $\mathbf{x}_2 = (0.19,\ 1.0)$.
3. Each coordinate evolves on its own. The second coordinate (curvature 10) reaches $1$ in one step,
   because $1 - 0.1 \cdot 10 = 0$. The first (curvature 1) follows $x_{k,1} = 1 - 0.9^k$.

The error $\mathbf{e}_k = \mathbf{x}_k - \mathbf{x}^*$ obeys $\mathbf{e}_{k+1} = (I - \eta A)\mathbf{e}_k$.
Here $I - \eta A$ is diagonal with entries $0.9$ and $0$. The steep direction is fixed in one step,
and the flat direction decays by $0.9$ per step.

## Why it works

For a quadratic, the update is linear in the error, so the whole behaviour is set by the eigenvalues
$\lambda_1 \le \dots \le \lambda_d$ of $A$. Write $A = Q\Lambda Q^\top$. In the eigenbasis each error
component is multiplied by $1 - \eta\lambda_i$ per step. Therefore:

- **Stability ($A \succ 0$):** all components shrink iff $|1 - \eta\lambda_i| < 1$ for every $i$, which is
  $0 < \eta < 2/\lambda_{\max}$. This is the classic bound (Boyd & Vandenberghe, Convex Optimization,
  chapter on unconstrained minimization). At exactly $\eta = 2/\lambda_{\max}$ the steepest error component
  flips sign forever without decaying.
- **Speed:** the worst factor is minimised at $\eta^* = 2/(\lambda_{\max}+\lambda_{\min})$, giving
  $\rho = (\kappa-1)/(\kappa+1)$ with $\kappa = \lambda_{\max}/\lambda_{\min}$.

A quadratic is a local model. A general smooth $f$ needs its own hypotheses, stated with $\nabla f$
$L$-Lipschitz (the curvature is at most $L$):

| Assumptions | Step | Guarantee |
|---|---|---|
| convex, minimiser $\mathbf{x}^\star$ exists | $0<\eta\le 1/L$ | $f(\mathbf{x}_k)-f^\star\le\lVert\mathbf{x}_0-\mathbf{x}^\star\rVert^2/(2\eta k)$, an $O(1/k)$ rate |
| also $\mu$-strongly convex | $\eta=1/L$ | $f(\mathbf{x}_k)-f^\star\le(1-\mu/L)^k\,(f(\mathbf{x}_0)-f^\star)$, a linear rate |
| nonconvex, bounded below | $0<\eta<2/L$ | $f$ decreases each step and $\lVert\nabla f(\mathbf{x}_k)\rVert\to 0$: a stationary point only |

Locators for these bounds are per the Library note: Boyd & Vandenberghe §9.3 (smooth-descent baseline)
and Bubeck §3.2 (gradient bounds). [source: library-refresh/01-Mathematics/08-Optimization/2026 -- Library -- Proximal Steps and Accelerated Convex Optimization -- Concept Note.md]

## Signal vs. noise

- **Signal:** the update rule, the stability bound $2/\lambda_{\max}$, and the fact that
  conditioning, not the iteration count, decides how long training takes.
- **Signal:** every convergence claim names its assumptions: smoothness, convexity and the step size.
- **Signal:** count steps with the stop test before the update. Off-by-one bugs here are common.
- **Noise for now:** line-search details (a line search costs extra function evaluations per step), momentum and Adam. They change the update,
  not the stability idea.

## Where it breaks

- **Too large a step.** Between $1/\lambda_{\max}$ and $2/\lambda_{\max}$ the error oscillates in
  sign while shrinking. Counterexample: $f(x) = \tfrac{L}{2}x^2$ with $L=10$ and $x_0=1$. At
  $\eta=1.9/L$ the iterates are $(-0.9)^k$, so $x_{50}\approx 0.005$. At $\eta=2/L$ they alternate
  $\pm1$ forever. At $\eta=2.1/L$ they are $(-1.1)^k$, so $x_{10}\approx 2.59$ and $x_{50}\approx 117$.
  A factor of magnitude $1.1$ overflows float64 after about $7{,}400$ steps. Detect the first
  non-finite iterate and stop.
- **Zig-zag.** On a long thin valley the gradient points across it, not toward the minimum. The path
  alternates sides and crawls.
- **Scale-dependent stopping.** Use a relative test $\lVert\nabla f(\mathbf{x}_k)\rVert \le
  \varepsilon\lVert\nabla f(\mathbf{x}_0)\rVert$ when the objective's scale is unknown. Do not stop on a small step alone: it may come from a small $\eta$.
- **Non-convex functions.** The stop test certifies only stationarity. For $f(x)=\cos x$, which is
  $1$-smooth, starting at $x_0=0$ the gradient is zero, so the rule stops after $0$ updates at $x=0$,
  which is a global maximum. From $x_0=0.1$ the iterates approach $\pi$, a minimum.

## Interview angle

"Walk me through gradient descent, and what goes wrong when the learning rate is too large." Answer
shape: the update, then the stability bound $2/\lambda_{\max}$ for a quadratic (divergence beyond
it, and oscillation just above $1/\lambda_{\max}$), then conditioning as the reason training
takes many steps, and the stopping rule with a non-finite check. Follow-ups: why $2/\lambda$, and what
to do when $\kappa$ is large. Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/partial-derivatives-and-gradients` (the gradient), `cml/eigenvalues-and-eigenvectors`
  (the stability bound).
- **Unlocks:** `cml/stochastic-and-mini-batch-gradient-descent` (the same update with a noisy gradient),
  `cml/normal-equation` (a closed-form solve of the same quadratic), `cml/feature-scaling` (scaling reduces $\kappa$).
- **Lineage:** Cauchy's 1847 note on simultaneous equations is usually credited as the origin of steepest descent, the ancestor of this update. The learning-rate
  bound on quadratics is the standard analysis from the optimisation literature.
