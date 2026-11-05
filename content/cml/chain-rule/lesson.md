---
title: Chain rule
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

If $z = g(u)$ and $u = f(x)$, then the derivative of the composition is the product of the local derivatives along the path: $\dfrac{dz}{dx} = g'(f(x))\,f'(x)$.

For vector links, each local derivative is a Jacobian, and composition multiplies them in link order: $J_{g\circ f}(\mathbf{x}) = J_g(f(\mathbf{x}))\,J_f(\mathbf{x})$ ([Jacobians note](../../../sources/library-refresh/01-Mathematics/03-Calculus/2026%20--%20Library%20--%20From%20Derivatives%20to%20Jacobians%20and%20Hessians%20--%20Concept%20Note.md)). Reverse mode multiplies from the output end on a row vector $\bar{\mathbf{y}}^\top$; forward mode multiplies from the input end on a tangent $\dot{\mathbf{x}}$ ([Baydin et al. 2018, §3.1–3.2](../../../sources/library-refresh/01-Mathematics/03-Calculus/2018%20--%20Baydin%20et%20al%20--%20Automatic%20Differentiation%20as%20Executable%20Chain%20Rules%20--%20Study%20Note.md)). Most links are elementwise, so their Jacobians are diagonal; the linear link is the exception.

The model is one sigmoid unit trained on squared error. With $n$ samples in rows of $X\in\mathbb{R}^{n\times d}$, targets $\mathbf{y}\in\mathbb{R}^n$ and weights $\mathbf{w}\in\mathbb{R}^d$:

| Link | Expression | Local Jacobian | Local derivative |
|---|---|---|---|
| Linear | $\mathbf{z} = X\mathbf{w}$ | $(n,d)$, equal to $X$ | row $i$ is $\mathbf{x}_i^\top$ |
| Sigmoid | $\mathbf{s} = \sigma(\mathbf{z})$, $\sigma(z) = 1/(1+e^{-z})$ | $(n,n)$, diagonal | $\sigma'(z) = s(1-s)$ |
| Square and mean | $L = \dfrac{1}{n}\sum_i (s_i - y_i)^2$ | gradient $(n,)$ | $\partial L/\partial s_i = 2(s_i-y_i)/n$ |

Reverse mode applies the transposed factors:

$$\nabla_{\mathbf{w}} L = X^\top\Big[\tfrac{2}{n}(\mathbf{s}-\mathbf{y}) \odot \mathbf{s} \odot (\mathbf{1}-\mathbf{s})\Big], \qquad (d,)\ \leftarrow\ (d,n)\,(n,).$$

The transpose is not the derivative of the linear link, which is $X$. It appears because the backward pass sends the upstream gradient back through the link. For $Z=XW$ the rule is $\nabla_W f = X^\top G$ ([Matrix note, linear-layer identity](../../../sources/library-refresh/01-Mathematics/05-Linear-Algebra/2012%20--%20Petersen%20and%20Pedersen%20--%20Matrix%20Differentials%20for%20Machine%20Learning%20--%20Study%20Note.md)).

## Worked example

One sample, $x = 1.5$, $y = 0$, $w = 0$. Then $z = 0$, $s = 0.5$, and $L = (s-y)^2$:

| Factor | Value | Why |
|---|---|---|
| Outer: $\partial L/\partial s = 2(s-y)$ | $1.0$ | square at $s = 0.5$ |
| Local: $\sigma'(z) = s(1-s)$ | $0.25$ | $\sigma'(0) = 1/4$, the largest value the sigmoid derivative takes |
| Inner: $\partial z/\partial w = x$ | $1.5$ | the linear link |

Product: $1.0 \times 0.25 \times 1.5 = 0.375$. Forget the middle factor and you get $1.5$, four times too large.

Two samples, $X = (1, 2)^\top$, $\mathbf{y} = (1, 0)$, $w = 0$: residuals $(-0.5, 0.5)$, so $\frac{2}{2}(1\cdot(-0.125) + 2\cdot 0.125) = 0.125$.

**Shared value.** Let $u = 2x$ and $L = u^2 + 3u$ at $x = 1.5$, so $u = 3$. Its adjoint sums both uses: $2u = 6$ from the square and $3$ from the linear term, so $\partial L/\partial u = 9$. Times $\partial u/\partial x = 2$ gives $18$, which autograd also returns.

## Why it works

Each link needs only its own local derivative, so one backward sweep costs a small constant multiple of the forward pass, however many weights there are. Nodes hold values, edges hold local derivatives, and the backward pass multiplies them and adds at shared nodes ([Baydin et al. 2018, §3.2](../../../sources/library-refresh/01-Mathematics/03-Calculus/2018%20--%20Baydin%20et%20al%20--%20Automatic%20Differentiation%20as%20Executable%20Chain%20Rules%20--%20Study%20Note.md)). Forward mode needs one pass per input direction, so reverse mode wins for a scalar loss.

**Vanishing gradients.** The sigmoid's local derivative is at most $0.25$ and decays as $e^{-|z|}$. Ten sigmoid links in series shrink the gradient reaching the first layer by at most $0.25^{10}\approx 9.5\times10^{-7}$, counting only the sigmoid factors. Saturated units pass almost nothing back: at $z=\pm 800$ the gradient is exactly zero.

**Stability.** Compute the sigmoid from $e=e^{-|z|}$, never from $e^{z}$ with large positive $z$. Use $\sigma'=e/(1+e)^2$, because $s(1-s)$ loses digits. At $z=30$ the true $\sigma'$ is $9.358\times10^{-14}$ and $s(1-s)$ gives $9.348\times10^{-14}$, a $0.1\%$ error. At $z=36$ the error is about $4\%$, and at $z=37$ the product is exactly $0$ while the true value is $8.5\times10^{-17}$.

Finite differences cannot check this regime. For $L=\sigma(z)^2$ at $z=30$ the true gradient is $1.9\times10^{-13}$, yet central differences with $h=10^{-6}$ return exactly $0$, since the two losses differ by about $4\times10^{-19}$, below float64 resolution near 1.

## Signal vs. noise

- **Signal:** the chain rule is a product of local Jacobians, one per link, in link order. Write the path before the formula.
- **Signal:** shapes. A stray `(n, 1)` can broadcast to an $(n,n)$ matrix without any error message.
- **Signal:** with a sigmoid output, squared error multiplies by $\sigma'$, and cross-entropy cancels it: its gradient is $(\sigma(z)-y)\,\mathbf{x}$.
- **Noise for now:** the general Jacobian calculus (`dl/jacobians-and-matrix-calculus`) and engine internals.

## Where it breaks

- **Silent saturation.** A confidently wrong sigmoid unit gets almost no gradient under squared error, so training stalls with a flat loss.
- **Overflow in the naive sigmoid.** $e^{z}/(1+e^{z})$ gives $\infty/\infty = \text{NaN}$ at $z = 710$. In $1/(1+e^{-z})$ the exponent overflows below $z \approx -709.8$ and the function returns $0$; at $z = -710$ the true value is about $4.5\times10^{-309}$.
- **Shapes.** `s - y[:, None]` gives an $(n,n)$ array. Using $X$ for $X^\top$ is an error when $n\neq d$, and silently wrong when they match.

## Interview angle

*60-second shape:* a gradient through a composition is a product of local Jacobians along the path, summed over paths. For the sigmoid unit it is $\frac{2}{n}X^\top[(s-y)\odot s\odot(1-s)]$: the outer square, the sigmoid slope, and the linear map's adjoint $X^\top$. The slope vanishes when the unit saturates, which is the failure to name.

*Likely follow-ups:* why cross-entropy avoids the vanishing factor, and why reverse mode is the right direction for a scalar loss.

## Connections

- **Prerequisites:** `cml/derivatives` (the one-variable derivative), `cml/partial-derivatives-and-gradients` (gradients of scalar functions of many variables).
- **Unlocks:** `cml/sigmoid-and-log-odds`, `cml/logistic-regression`, `dl/backprop-through-a-chain-of-scalars`, `dl/backprop-through-a-dense-layer`, `dl/reverse-mode-autograd-engine`.
- **Lineage:** the chain rule is standard calculus. Reverse-mode automatic differentiation has its history in the origins section of the Library's survey note ([Baydin et al. 2018, §3.3](../../../sources/library-refresh/01-Mathematics/03-Calculus/2018%20--%20Baydin%20et%20al%20--%20Automatic%20Differentiation%20as%20Executable%20Chain%20Rules%20--%20Study%20Note.md)); check it before making attribution claims.
