# Dominated Convergence and Gradients of Expectations

**Scope:** a modern bridge from Lebesgue integration to population-risk gradients. Axler and Lebl supply mathematical exposition; Bottou, Curtis, and Nocedal supply the optimization setting. This note does not assign historical priority to these textbooks.

A population loss is an integral, $R(\theta)=\mathbb E[\ell(\theta,Z)]$. Training usually differentiates a sampled loss and averages the resulting gradients. To connect that computation to the population objective, one must justify
$$
\nabla_\theta\mathbb E[\ell(\theta,Z)]
=\mathbb E[\nabla_\theta\ell(\theta,Z)].
$$
The equality is a theorem with hypotheses. Pointwise existence of derivatives is insufficient.

The dominated convergence theorem states: on a measure space, if measurable $f_k\to f$ almost everywhere and $|f_k|\le g$ almost everywhere for one integrable nonnegative function $g$, then $\int f_k\to\int f$. The same envelope must work across the sequence. Neither pointwise convergence nor finite integrals for each individual function supplies that control.

A simple failure is
$$
f_k(x)=k\,\mathbf1_{(0,1/k)}(x),\quad x\in(0,1).
$$
Every fixed positive $x$ eventually leaves the shrinking interval, so $f_k(x)\to0$. Yet $\int_0^1f_k(x)\,dx=1$. Mass concentrates where the pointwise limit no longer sees it. This example is a useful warning when a sequence of models develops rare but increasingly large losses.

Differentiation is itself a limit. For scalar $\theta$, apply dominated convergence to
$$
q_h(z)=\frac{\ell(\theta+h,z)-\ell(\theta,z)}h.
$$
One sufficient local setup is: $\ell(\theta_0,\cdot)$ is integrable; for almost every $z$, $\ell(\cdot,z)$ is differentiable throughout an interval around $\theta_0$; derivatives are measurable; and $|\partial_\theta\ell(\theta,z)|\le G(z)$ throughout that interval, with $\mathbb E G<\infty$. The mean value theorem bounds the difference quotients by $G$, and dominated convergence gives the desired equality. Vector parameters can be handled coordinate by coordinate under corresponding local envelopes.

For squared prediction error,
$$
\ell(\theta,(X,Y))=\tfrac12(\theta X-Y)^2,\qquad
\partial_\theta\ell=X(\theta X-Y).
$$
On $|\theta|\le M$, the derivative is bounded by $MX^2+|XY|$. Finite second moments of $X$ and $Y$ make this envelope integrable by Cauchy–Schwarz. Hence
$$
R'(\theta)=\theta\,\mathbb E[X^2]-\mathbb E[XY].
$$
The assumptions reveal exactly what could fail with heavy-tailed data.

The distribution in this calculation is fixed. If $Z$ itself has a parameter-dependent law, differentiating only the integrand generally omits another contribution. Reparameterization and score-function estimators are ways to address that different problem, each requiring its own justification. Likewise, differentiability almost everywhere in the data variable must not be silently exchanged for the existence of a common neighborhood and a common dominating function.

This closes a practical gap between [multivariable calculus](../03-Calculus/2026%20--%20Library%20--%20From%20Derivatives%20to%20Jacobians%20and%20Hessians%20--%20Concept%20Note.md) and [stochastic gradient theory](../08-Optimization/2018%20--%20Bottou%2C%20Curtis%20and%20Nocedal%20--%20Stochastic%20Gradients%2C%20Noise%20and%20Computation%20--%20Study%20Note.md). Lebesgue integration provides expectations; convergence control permits a derivative to pass through them; sampling can then estimate the justified gradient. These are conceptual dependencies. A gradient estimator may be unbiased while still having unbounded variance, and a correct population derivative does not by itself prove that an iterative optimizer converges.

## Sources and locators

- Axler, Measure, Integration & Real Analysis, §3B, theorem 3.31, printed pp.92–93 (PDF pp.107–108), gives dominated convergence. [Author's open-access edition](https://measure.axler.net/), revised June 12, 2026; original Springer publication 2020.
- Lebl, Basic Analysis II, version 6.3, §9.1, printed pp.65–70, presents differentiation under an integral and counterexamples. [Author edition](https://www.jirka.org/ra/realanal2.pdf). The local-envelope argument above is the library's application of these standard tools to expectation.
- Bottou, Curtis, and Nocedal, Optimization Methods for Large-Scale Machine Learning, §3 expected/empirical risk and §4.1 moment assumptions; equation (4.1), manuscript p.21, distinguishes population and finite-sample objectives. [Inspected June 23, 2018 author manuscript](https://leon.bottou.org/publications/pdf/tr-optml-2016.pdf).

