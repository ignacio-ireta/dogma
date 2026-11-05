# Augustin-Louis Cauchy - Steepest Descent (1847)

#person #primary-source #milestone #optimization #gradient-descent #lineage

**Person:** Augustin-Louis Cauchy (1789-1857)  
**Work:** "Methode generale pour la resolution des systemes d'equations simultanees"  
**Date:** 1847  
**Primary source:** Cauchy's three-page note in *Comptes Rendus Hebdomadaires des Seances de l'Academie des Sciences*, translated by Richard J. Pulskamp  
**Paper:** `corpus/raw/library-papers/cauchy-steepest-descent-1847.pdf`  
**Reference pages:** Pulskamp translation; original citation in *Comptes Rendus* 25, 536-538; Oeuvres completes I, 10, 399-402

## The Problem That Made This Necessary

By the middle of the nineteenth century, calculus could describe change, Lagrange could keep constraints inside analytic mechanics, and least squares had made residual error mathematically respectable. But a very ordinary computational problem still resisted the older toolbox:

```text
How do you solve a large system of equations when symbolic elimination becomes impossible?
```

Cauchy opens the paper by naming exactly that failure. The usual method was to eliminate variables until one equation remained. But elimination could fail outright, and when it did work, the resulting equation could be uglier than the original system.

That is the pressure that made steepest descent necessary. Mathematics needed a way to make progress without solving the whole problem symbolically in one heroic rearrangement.

Cauchy's answer is the ancestor of the modern training loop:

```text
measure how wrong you are -> compute a local direction of improvement -> move -> repeat
```

## Key Ideas

### 1. Replace exact solution with decreasing error

Cauchy starts with a nonnegative continuous function:

```text
u = f(x, y, z, ...)
```

Solving `u = 0` can be treated as making `u` decrease until it vanishes, or at least until it reaches a minimum. This is the conceptual turn:

```text
root-finding becomes minimization
```

The old question was "can I solve these equations exactly?" Cauchy's question is "can I define a quantity that gets smaller when I improve the approximation?"

### 2. Partial derivatives point downhill

Let `X, Y, Z, ...` be the partial derivatives of `u` with respect to `x, y, z, ...`. For small changes `alpha, beta, gamma, ...`, Cauchy writes the first-order approximation:

```text
u_new ~= u + alpha X + beta Y + gamma Z + ...
```

Then he chooses:

```text
alpha = -theta X
beta  = -theta Y
gamma = -theta Z
```

so that:

```text
u_new ~= u - theta (X^2 + Y^2 + Z^2 + ...)
```

For sufficiently small positive `theta`, this decreases the objective. In modern notation:

```text
x_{k+1} = x_k - theta grad f(x_k)
```

That is steepest descent in its essential form. The point is not that the notation is modern. It is not. The point is that the logic is already there: local derivatives supply a direction that should reduce the scalar error.

### 3. Step size is a separate problem

Cauchy does not stop at "move opposite the derivatives." He also considers how to choose `theta`, the amount of movement along that direction.

He proposes increasing `theta` until the value along that line reaches a minimum, represented by:

```text
D_theta Theta = 0
```

That anticipates line search. Direction and step length are separate questions from the very beginning of gradient descent.

This is still one of optimization's permanent bargains. Too small a step wastes time. Too large a step overshoots or diverges. Every learning-rate schedule, backtracking line search, trust-region method, and adaptive optimizer lives under that old pressure.

### 4. Many equations become one scalar loss

For a system:

```text
u = 0, v = 0, w = 0, ...
```

Cauchy reduces the system to one nonnegative objective:

```text
u^2 + v^2 + w^2 + ... = 0
```

This is the quiet bridge to machine learning. Many mismatches become one scalar loss. The paper even allows the number of equations to exceed the number of unknowns, the shape we now recognize in overdetermined least-squares problems and data fitting.

This is not merely a numerical trick. It makes "wrongness" differentiable and therefore actionable.

### 5. Descent is local, not omniscient

Cauchy is careful about the limits of the method. The procedure converges toward a minimum, and reaching an actual zero depends on choosing starting values conveniently.

That honesty matters for the lineage. Gradient descent is not magic. It follows local information on a landscape. If the landscape has poor conditioning, bad minima, flat regions, saddle points, or noisy estimates, the method needs repair.

Those repairs become much of modern optimization.

## Historical Context

Cauchy was writing after the calculus revolution, after Lagrange's analytic mechanics, and after Gauss had made least-squares fitting central to scientific computation. The mathematical pieces existed separately:

- differential calculus could describe local change,
- least squares could compress residuals into one error measure,
- numerical approximation could improve estimates iteratively.

What Cauchy's 1847 paper does is combine those pieces into a general method for simultaneous equations. He does not write a modern ML optimizer, but he creates the recognizable skeleton:

```text
scalar objective + derivatives + iterative update
```

The paper is short because the move is simple. The consequences are not.

## Lineage

```text
Fermat (local extrema by perturbation)
  -> Newton and Leibniz (calculus of rates and accumulation)
      -> Euler 1744 (optimization over curves/functions)
          -> Lagrange 1788 (constraints stay in the equations)
              -> Cauchy 1847 (walk downhill by derivative information)
                  -> Robbins-Monro 1951 (descent under noisy observations)
                      -> SGD, backpropagation, momentum, Adam, AdamW
```

**Built on:** [[../04-Early-Calculus/Fermat - Adequality and Tangents 1637|Fermat]], because extrema had to become a local-change problem; [[../05-Newton-Leibniz/Newton-Leibniz - Fundamental Theorem of Calculus|Newton and Leibniz]], because the method depends on differentiating a changing quantity; [[Lagrange - Analytical Mechanics and Multipliers 1788|Lagrange]], because optimization had become an analytic language for systems rather than isolated tricks; and Gauss's least-squares worldview, because many residuals could be treated as one error measure.  
**Enables:** numerical optimization, line search, least-squares iteration, [[../../AI-ML/03-Optimization/Robbins Monro Stochastic Approximation 1951|Robbins-Monro stochastic approximation]], stochastic gradient descent, backpropagation, and adaptive optimizers.  
**Later echoes:** learning rates, minibatch gradients, momentum, adaptive second moments, loss landscapes, and every neural-network training loop that reduces a scalar loss.

## Why It Matters for Mathematics

Cauchy changes the role of calculus in computation.

Calculus had already found exact derivatives and exact extrema. Cauchy uses derivatives as instructions for movement when exact solution is not available.

That is a different kind of mathematics:

```text
not just prove what the solution is,
but design a process that approaches it
```

This process view is the bridge from classical analysis into numerical analysis and optimization theory. It makes convergence, conditioning, step size, approximation error, and stopping criteria unavoidable mathematical objects.

## Why It Matters for AI

Modern AI is Cauchy's move at absurd scale.

A neural network usually cannot solve its weights by symbolic elimination. It builds a scalar loss from many errors, differentiates that loss with respect to millions or billions of parameters, and updates the parameters to make the loss smaller.

Backpropagation answers the credit-assignment question: how do we compute all those derivatives efficiently? But Cauchy supplies the older optimization question:

```text
once you have the derivatives, what do you do with them?
```

The answer is still recognizable:

```text
move parameters in a direction expected to reduce loss
```

Robbins-Monro was needed because ML gradients are often sampled and noisy. Momentum was needed because plain descent forgets its previous motion. AdaGrad, RMSProp, Adam, and AdamW were needed because different parameters experience different scales, sparsity, curvature, and regularization pressure. But all of those refinements preserve Cauchy's skeleton.

This is why the note belongs after Lagrange in the Mathematics lineage. Euler and Lagrange make optimization conceptually large enough. Cauchy makes it algorithmic.

## References

- Cauchy, A.-L. "Methode generale pour la resolution des systemes d'equations simultanees." *Comptes Rendus Hebdomadaires des Seances de l'Academie des Sciences* 25, 536-538, 1847. `corpus/raw/library-papers/cauchy-steepest-descent-1847.pdf`
- Pulskamp, Richard J., trans. English translation of Cauchy's 1847 note, July 17, 2010. Included in the local PDF above.
- Cauchy, A.-L. *Oeuvres completes*, series I, volume 10, 399-402.

## Status

- [x] Source PDF present in `corpus/raw/library-papers/cauchy-steepest-descent-1847.pdf`
- [x] PDF verified with `file` as a PDF document
- [x] Note created as the bridge from analytical optimization to gradient descent
- [x] Cross-linked to AI/ML optimization lineage

---

*Cauchy's quiet revolution is that failure to solve exactly is not failure. It is an invitation to measure error and move.*
