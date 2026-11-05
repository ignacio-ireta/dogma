# Cauchy — Steepest Descent (1847)

*Optimization begins when an impossible equation is replaced by a quantity that can be made smaller.*

**← [[../_MOC|AI/ML Lineage]]**

---

## Paper / Year

**"Méthode générale pour la résolution des systèmes d'équations simultanées"** — Augustin-Louis Cauchy, *Comptes Rendus Hebdomadaires des Séances de l'Académie des Sciences*, 1847.

- Paper: `corpus/raw/library-papers/cauchy-steepest-descent-1847.pdf`
- Source: English translation by Richard J. Pulskamp of the original three-page paper
- Local PDF verified as a PDF document

## Key Ideas

1. **Stop insisting on symbolic elimination.**  
   Cauchy begins with a practical failure. A system of simultaneous equations could sometimes be reduced to one equation by eliminating variables, but elimination might be impossible or produce an equation more complicated than the original system. The old method tried to solve the whole structure exactly before making progress.

2. **Turn equation solving into minimization.**  
   For an equation \(u=f(x,y,z,\ldots)=0\), Cauchy assumes \(u\) is continuous and nonnegative. Solving the equation can then be reframed as repeatedly decreasing \(u\) until it reaches zero, or at least a minimum.

3. **The derivatives identify a local downhill direction.**  
   Let \(X, Y, Z,\ldots\) be the partial derivatives of \(u\). For small changes \(\alpha,\beta,\gamma,\ldots\), the first-order approximation is:

   ```text
   u_new ≈ u + αX + βY + γZ + ...
   ```

   Cauchy chooses:

   ```text
   α = -θX,  β = -θY,  γ = -θZ, ...
   ```

   so that:

   ```text
   u_new ≈ u - θ(X² + Y² + Z² + ...)
   ```

   For a sufficiently small positive \(\theta\), the objective decreases. In modern notation, this is the gradient step \(x_{k+1}=x_k-\theta\nabla f(x_k)\).

4. **Step size is already the central bargain.**  
   Cauchy does not merely say "follow the negative gradient." He considers varying \(\theta\) until the objective along that direction reaches a minimum, which anticipates exact line search. He also gives a small-step approximation when the residual is already small. Direction and step length are separate problems from the beginning.

5. **Many equations become one least-squares objective.**  
   For a system:

   ```text
   u = 0,  v = 0,  w = 0, ...
   ```

   Cauchy replaces it with:

   ```text
   u² + v² + w² + ... = 0
   ```

   This is the crucial bridge: a collection of residuals becomes one scalar loss. The paper even allows more equations than unknowns, the shape later called an overdetermined least-squares problem.

6. **It is a local method, not magic.**  
   Cauchy says the process converges toward a minimum and notes that reaching a zero depends on choosing the starting values conveniently. A stationary point need not solve the original equations. The method creates a general route downhill, but not a guarantee that every landscape leads to the desired destination.

## What Came Before → What It Enabled

### What came before

- Systems of equations were attacked through elimination or problem-specific algebra.
- Newton-style methods could converge rapidly near a solution, but required solving a local linear system and starting from an already useful approximation.
- Least squares had supplied the idea of measuring residual error, but not yet the general iterative rule that moves parameters by local derivative information.

### What broke

Exact algebra did not scale.

Elimination could fail outright, and even when it worked, intermediate expressions could become more difficult than the system being solved. Newton's method was powerful near a root but demanded curvature-like information and a good starting point. They could not keep treating every numerical problem as if a closed-form rearrangement were waiting to be discovered.

Cauchy's answer was to stop asking for the solution all at once. Build a scalar measure of wrongness, inspect how it changes locally, and take a step that reduces it.

### What it enabled

- [[Robbins Monro Stochastic Approximation 1951|Robbins-Monro]] asks what happens when the exact function or direction is hidden behind noisy observations.
- [[Widrow ADALINE 1960|Widrow-Hoff / LMS]] turns squared prediction error into an online weight update for an adaptive neuron.
- [[Polyak Heavy-Ball Momentum 1964|Polyak momentum]] repairs steepest descent's amnesia and slow zig-zagging on ill-conditioned objectives.
- [[Rumelhart Hinton Williams Backpropagation 1986|Backpropagation]] makes the needed gradients available in multilayer networks.
- AdaGrad, Adam, AdamW, Muon, and the rest of the modern optimizer stack all preserve Cauchy's basic decomposition: define a loss, estimate a direction, choose an update, repeat.

## Why It Matters

Cauchy's paper contains the conceptual skeleton of machine learning training.

A model does not usually solve its parameters by symbolic elimination. It converts mismatches into a scalar loss, differentiates that loss with respect to many parameters, and changes those parameters in a direction expected to reduce the error. Even the most elaborate optimizer still begins from that bargain.

The paper also explains why the optimizer lineage did not end in 1847.

Cauchy's direction is exact and deterministic. Machine learning sees noisy minibatches, so [[Robbins Monro Stochastic Approximation 1951|Robbins-Monro]] was needed. Plain descent forgets its previous motion, so [[Polyak Heavy-Ball Momentum 1964|momentum]] was needed. One global step size behaves badly across sparse and uneven coordinates, so AdaGrad and Adam were needed. Weight decay interacts incorrectly with adaptive scaling, so AdamW was needed.

Each later breakthrough repairs something Cauchy's clean setup cannot survive at modern scale. But the basic move remains his: when direct solution breaks, turn the problem into a surface and walk downhill.

## Lineage

- **Built on:** differential calculus, least-squares residuals, numerical approximation, and the failure of symbolic elimination for large simultaneous systems.
- **Leads to:** [[Robbins Monro Stochastic Approximation 1951|stochastic approximation]], [[Widrow ADALINE 1960|LMS / ADALINE]], [[Polyak Heavy-Ball Momentum 1964|momentum]], stochastic gradient descent, backpropagation, and adaptive optimizers.
- **Modern echo:** every neural-network training loop still has Cauchy's three parts: a scalar loss, derivatives with respect to parameters, and a step chosen to make the loss smaller.

---

*Before machines could learn from error, mathematics needed a general way to move toward less error.*
