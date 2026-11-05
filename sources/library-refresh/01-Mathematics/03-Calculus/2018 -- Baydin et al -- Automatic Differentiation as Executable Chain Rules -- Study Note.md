# Automatic Differentiation as Executable Chain Rules

**Work:** Atılım Güneş Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind, *Automatic Differentiation in Machine Learning: a Survey* (JMLR, 2018). This survey explains the field and its history; it is not the original invention of automatic differentiation.

Knowing the chain rule does not by itself make a billion-parameter gradient affordable. A program reuses intermediate values, branches, and repeats operations. Differentiating a fully expanded symbolic expression can destroy that sharing; finite differences require choosing a perturbation and repeatedly evaluating the program. Automatic differentiation preserves the computation's elementary operations and propagates their local derivatives.

For $f:\mathbb R^n\to\mathbb R^m$, forward mode computes a Jacobian–vector product
$$
\dot y=J_f(x)\dot x.
$$
Each elementary operation carries its value and a tangent. For multiplication $v=ab$, the tangent rule is $\dot v=b\dot a+a\dot b$. The algebra of dual numbers, with $\epsilon^2=0$, packages the same rule:
$$
(a+\dot a\epsilon)(b+\dot b\epsilon)
=ab+(b\dot a+a\dot b)\epsilon.
$$
One seed direction gives one directional derivative; the full Jacobian generally requires multiple directions.

Reverse mode starts with an output sensitivity and accumulates
$$
\bar x=J_f(x)^\top\bar y.
$$
For a primitive $v=\phi(u_1,\ldots,u_r)$, it adds $\bar v\,\partial\phi/\partial u_i$ to each input's adjoint. The addition matters: several downstream uses of one value contribute to the same derivative. A forward evaluation records dependencies and needed values, then a reverse traversal propagates adjoints.

For example, let $u=x_1x_2$ and $y=\sin u+x_1^2$. With $\bar y=1$, the sine sends $\bar u=\cos u$; multiplication then contributes $x_2\cos u$ to $\bar x_1$ and $x_1\cos u$ to $\bar x_2$. The square adds another $2x_1$:
$$
\nabla y=(x_2\cos(x_1x_2)+2x_1,\;x_1\cos(x_1x_2)).
$$
This small example exposes the same reuse that makes backpropagation effective.

For a scalar output, reverse mode obtains all input derivatives with work proportional to a constant multiple of the executed primitive computation, under the usual cost model. Its storage can grow with the saved computation. Checkpointing trades recomputation for memory; forward mode can be preferable when inputs or requested directions are few. Mode choice depends on input/output dimensions and the derivative products actually needed.

AD applies analytic derivative rules to an executed program. It does not eliminate floating-point error, repair an unstable forward computation, or automatically establish that the modeled function is differentiable. At a branch boundary, differentiating the taken branch may not describe a derivative of the whole function. At a nonsmooth primitive, a chosen convention must be understood. Solver tolerances introduce further issues.

The prerequisite is [the derivative as a linear map](2026%20--%20Library%20--%20From%20Derivatives%20to%20Jacobians%20and%20Hessians%20--%20Concept%20Note.md); the immediate output is a cheap gradient for [sampled optimization](../08-Optimization/2018%20--%20Bottou%2C%20Curtis%20and%20Nocedal%20--%20Stochastic%20Gradients%2C%20Noise%20and%20Computation%20--%20Study%20Note.md). [ODE adjoints](2018%20--%20Chen%20et%20al%20--%20Neural%20ODEs%20and%20Continuous%20Adjoint%20Sensitivity%20--%20Study%20Note.md) extend sensitivity propagation to a continuous evolution, while [implicit differentiation](../08-Optimization/2022%20--%20Blondel%20et%20al%20--%20Implicit%20Differentiation%20and%20Optimization%20Layers%20--%20Study%20Note.md) differentiates a solution equation. These are related uses of calculus with different numerical objects. The historical Leibniz note supplies a conceptual predecessor; the survey's historical section should be consulted before asserting specific inventor or influence claims.

## Sources and locators

- Baydin et al., Automatic Differentiation in Machine Learning - A Survey, §1, pp.2–3 (differentiation methods); §2, pp.3–8 (comparison with numerical and symbolic differentiation); §3.1, pp.9–11 (forward mode and dual numbers); §3.2, pp.12–14 (reverse mode); §3.3, pp.13–14 (origins); §4, pp.14–21 (applications); §5, pp.21–26 (implementations). [Official JMLR article](https://jmlr.org/papers/v18/17-468.html), published April 2018. Page numbers are the article's printed pages.
- Lebl, Basic Analysis II, version 6.3, §8.3, printed pp.35–45, provides the underlying derivative and chain rule for multivariable maps. [Author source](https://www.jirka.org/ra/realanal2.pdf). This supplies modern rigorous exposition, not AD history.

