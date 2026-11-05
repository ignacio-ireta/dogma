---
title: Partial derivatives and gradients
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Let $f:\mathbb{R}^n\to\mathbb{R}$ be a scalar function of a vector $\mathbf{x}=(x_1,\dots,x_n)$.

| Object | Definition | Shape | Meaning |
|---|---|---|---|
| Partial derivative | $\dfrac{\partial f}{\partial x_i}$: differentiate in $x_i$, hold every other $x_j$ fixed | scalar | the slope along axis $i$ |
| Gradient | $\nabla f(\mathbf{x}) = \big(\partial f/\partial x_1, \dots, \partial f/\partial x_n\big)$ | $(n,)$, the shape of $\mathbf{x}$ | the vector of all partials |
| Directional derivative | $D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u}$, for a unit vector $\mathbf{u}$ | scalar | the slope in direction $\mathbf{u}$ |

Fix one layout before you differentiate. For $\mathbf{f}:\mathbb{R}^n\to\mathbb{R}^m$ the Jacobian is $m\times n$, and a scalar $f$ is its $m=1$ row. Texts differ on writing that row or the column $\nabla f$; transposed answers come from mixing them. The Frobenius convention in the Matrix note, $df=\langle\nabla f, d\mathbf{x}\rangle$, gives a gradient the shape of its variable ([Matrix note, convention section](../../../sources/library-refresh/01-Mathematics/05-Linear-Algebra/2012%20--%20Petersen%20and%20Pedersen%20--%20Matrix%20Differentials%20for%20Machine%20Learning%20--%20Study%20Note.md)).

The gradient is the direction of **steepest ascent**. Over unit vectors $\mathbf{u}$, $\nabla f\cdot\mathbf{u}$ is largest when $\mathbf{u}$ points along $\nabla f$, and then it equals $\lVert\nabla f\rVert$. Its negative is the steepest descent direction, which is what gradient descent follows. Along a level set $f$ is constant, so $\nabla f$ is orthogonal to it.

Closed forms you will use, with $\mathbf{x}\in\mathbb{R}^n$ and $A\in\mathbb{R}^{n\times n}$:

- $\nabla(\mathbf{a}^\top\mathbf{x}) = \mathbf{a}$
- $\nabla(\mathbf{x}^\top A\mathbf{x}) = (A + A^\top)\mathbf{x}$, which is $2A\mathbf{x}$ when $A$ is symmetric
- $\nabla \log\sum_i e^{x_i} = \operatorname{softmax}(\mathbf{x})$

## Worked example

Let $f(x_1,x_2) = x_1^2 + 3x_1x_2$ at $\mathbf{x} = (1, 2)$.

1. $\partial f/\partial x_1 = 2x_1 + 3x_2 = 2 + 6 = 8$
2. $\partial f/\partial x_2 = 3x_1 = 3$
3. $\nabla f(1,2) = (8, 3)$

Check with central differences at $h = 10^{-5}$: the $x_1$ partial gives $8.000000000008$, and the $x_2$ partial gives $3.000000000064$. Since $f$ is quadratic, the central difference is exact up to round-off.

Steepest ascent: $\lVert\nabla f\rVert=\sqrt{73}\approx 8.544$, against rates $8.0$ along $(1,0)$ and $3.0$ along $(0,1)$.

## Why it works

Each partial is a one-variable derivative, but all of them together do not guarantee a linear model. The model $f(\mathbf{x}+\mathbf{d}) = f(\mathbf{x}) + \nabla f\cdot\mathbf{d} + o(\lVert\mathbf{d}\rVert)$ needs $f$ to be *differentiable*, with one linear map valid in every direction. Take $f(0,0)=0$ and $f(x,y)=xy/\sqrt{x^2+y^2}$ elsewhere. Both partials at the origin are $0$, yet along $(t,t)$ the error divided by $\lVert(t,t)\rVert$ stays $1/2$ ([Jacobians note](../../../sources/library-refresh/01-Mathematics/03-Calculus/2026%20--%20Library%20--%20From%20Derivatives%20to%20Jacobians%20and%20Hessians%20--%20Concept%20Note.md)). Where $f$ is differentiable, Cauchy–Schwarz gives $\nabla f\cdot\mathbf{u}\le\lVert\nabla f\rVert$ for unit $\mathbf{u}$. Deisenroth, Faisal and Ong's *Mathematics for Machine Learning* covers partial derivatives and the gradient in §5.2.

A central difference per coordinate costs $2n$ evaluations, with $O(h^2)$ error per partial. The step is a trade-off: truncation error falls as $h$ shrinks, while round-off, about $\varepsilon\lvert f\rvert/h$, grows. For $f=e^x$ at $x=1$ (true derivative $e$), measured float64 errors:

| $h$ | forward error | central error |
|---|---|---|
| $10^{-2}$ | $1.4\times10^{-2}$ | $4.5\times10^{-5}$ |
| $10^{-4}$ | $1.4\times10^{-4}$ | $4.5\times10^{-9}$ |
| $10^{-6}$ | $1.4\times10^{-6}$ | $1.6\times10^{-10}$ |
| $10^{-8}$ | $6.6\times10^{-9}$ | $6.6\times10^{-9}$ |
| $10^{-10}$ | $1.5\times10^{-6}$ | $6.7\times10^{-7}$ |

Scanning $h$, the best forward step is about $1.3\times10^{-8}$ (error $2\times10^{-9}$), near $\varepsilon^{1/2}$. The best central step is about $5\times10^{-6}$ (error $10^{-12}$), near $\varepsilon^{1/3}\approx6\times10^{-6}$.

**Gradient checking** compares a hand-derived $\mathbf{g}_a$ with a numerical $\mathbf{g}_n$. One seeded run (`default_rng(0)` per function, 200 points each: quadratic forms with $n=1$–$5$ and $x\in[-3,3]^n$; log-sum-exp with $n=2$–$5$ on $[-5,5]^n$; Rosenbrock with $n=2$–$4$ on $[-1.5,1.5]^n$; steps $\varepsilon^{1/3}$ central and $\varepsilon^{1/2}$ forward, each times $\max(1,|x_i|)$) gives relative error $\lVert\mathbf g_a-\mathbf g_n\rVert/\max(1,\lVert\mathbf g_n\rVert)$: for central differences, median $2\times10^{-11}$ and worst $5\times10^{-10}$; for forward differences, median $1.5\times10^{-8}$ and worst $1.7\times10^{-7}$. A sign error or a missing factor gives an error of order one, far above either. Check away from kinks, where a correct formula can still disagree.

## Signal vs. noise

- **Signal:** the gradient is a vector of partials with the shape of $\mathbf{x}$, and it points uphill. Descent uses its negative.
- **Signal:** check gradients with central differences in float64, on small random inputs.
- **Noise for now:** the full Jacobian calculus (`dl/jacobians-and-matrix-calculus`) and second derivatives (`cml/hessians-definiteness-and-critical-points`).

## Where it breaks

- **Cost.** $2n$ evaluations is impossible for millions of weights. Backprop gives the exact gradient in a small constant multiple of one forward pass ([Baydin et al. 2018, §3.2](../../../sources/library-refresh/01-Mathematics/03-Calculus/2018%20--%20Baydin%20et%20al%20--%20Automatic%20Differentiation%20as%20Executable%20Chain%20Rules%20--%20Study%20Note.md)).
- **Scale.** The step must scale with each coordinate. For $\lVert\mathbf{x}\rVert^2$ at $(10^8, 1)$, an absolute step of $\varepsilon^{1/3}\approx 6\times10^{-6}$ gives relative error $9\times10^{-4}$ on the first partial, and the scaled step gives $1.5\times10^{-11}$. Round-off follows the size of $f$ ($\approx10^{16}$ here), so the second partial (true value 2) is near $1.65\times10^5$ with either step. Rescale.
- **Float32.** The central step is about $\varepsilon_{32}^{1/3}\approx 4.9\times10^{-3}$. One seeded run (`default_rng(0)`, 2000 points for $\sin$ on $[0.2, 2.5]$, step $\varepsilon_{32}^{1/3}\max(1,|x|)$ in float32) gives median relative error $4.4\times10^{-6}$ and worst $2.7\times10^{-5}$. Too coarse for a gradient check; check in float64.
- **Zero gradient is not a minimum.** It can be a saddle or a maximum; second derivatives tell them apart.

## Interview angle

*60-second shape:* a partial differentiates along one coordinate with the others frozen. The gradient stacks the partials, with the shape of $\mathbf{x}$. For a differentiable $f$ it points uphill, so descent moves along its negative. Backprop computes it in a small constant multiple of one forward pass. Central differences are the standard check.

*Likely follow-ups:* why the gradient is orthogonal to level sets, and why numerical gradients are used for checking but not for training.

## Connections

- **Prerequisites:** `cml/derivatives` (the one-variable difference quotient), `cml/vector-operations` (the dot product in the directional derivative).
- **Unlocks:** `cml/chain-rule` (gradients of compositions), `cml/gradient-descent` (stepping along $-\nabla f$), `dl/jacobians-and-matrix-calculus`, `dl/gradients-with-framework-autograd`.
- **Lineage:** Cauchy's 1847 note on simultaneous equations is usually credited as the origin of steepest descent.
