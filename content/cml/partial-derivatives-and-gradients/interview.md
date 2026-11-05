# Interview notes: Partial derivatives and gradients

## Q1. What is a gradient, and why does it point in the direction of steepest ascent?

**60-second answer.** For a scalar function of many variables, the gradient is the vector of partial derivatives, one per coordinate. Each partial differentiates along one axis with the others held fixed. For a differentiable $f$, the linear approximation $f(\mathbf{x}+\mathbf{d}) \approx f(\mathbf{x}) + \nabla f\cdot\mathbf{d}$ shows the direction: among unit vectors, $\nabla f\cdot\mathbf{u}$ is largest when $\mathbf{u}$ is parallel to $\nabla f$, by Cauchy–Schwarz. Gradient descent steps along the negative gradient for that reason.

**Follow-ups.**
- *Why is the gradient orthogonal to level sets?* Along a level set $f$ does not change, so the tangent direction $\mathbf{t}$ satisfies $\nabla f\cdot\mathbf{t}=0$.
- *What is the cost of computing it?* Central differences need $2n$ evaluations. Backprop needs a small constant multiple of one forward pass, which is why it is used in training.
- *What shape is the Jacobian of $\mathbb{R}^n\to\mathbb{R}^m$?* $m\times n$. A scalar output is the $m=1$ row, and the gradient is its transpose, with the shape of $\mathbf{x}$.

## Q2. How would you check a gradient you derived by hand?

**60-second answer.** Compare it with a central finite-difference gradient at a few random points, in float64. Use the relative error $\lVert\mathbf{g}_a-\mathbf{g}_n\rVert/\max(1,\lVert\mathbf{g}_n\rVert)$. A correct formula agrees to about $10^{-8}$ or better with central differences, and a sign error or a missing term shows up as an error of order one. Use central differences: their error is $O(h^2)$, while a forward difference has $O(h)$ error and typically agrees only to about $10^{-8}$, which can hide a small bug. Run the check on small inputs, since it costs $2n$ evaluations, and never in the training loop.

**Follow-ups.**
- *Why not check in float32?* The best central step is about $5\times10^{-3}$, and the error is about $10^{-5}$, which is too coarse to see subtle bugs.
- *What if the function has a kink?* The check can disagree at the kink even when the formula is right. Test away from it.

## Common mistakes

- Perturbing all coordinates at once and calling the result a partial derivative. That gives a directional derivative along $(1,\dots,1)$.
- Forgetting that $\nabla(\mathbf{x}^\top A\mathbf{x}) = (A+A^\top)\mathbf{x}$, and writing $2A\mathbf{x}$ for a non-symmetric $A$.
- Mutating the input array while perturbing one coordinate and forgetting to restore it.
- Reading a zero gradient as a minimum without checking the second derivatives.
