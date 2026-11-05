# Joseph-Louis Lagrange — Analytical Mechanics and Multipliers (1788)

#person #primary-source #milestone #optimization #constraints #calculus-of-variations #lineage

**Person:** Joseph-Louis Lagrange (1736-1813)  
**Work:** *Méchanique analitique*  
**Date:** 1788  
**Primary source:** Lagrange's 1788 treatise, digitized by Smithsonian Libraries and hosted through Internet Archive  
**Paper:** `corpus/raw/library-papers/lagrange-analytical-mechanics-1788.pdf`  
**Source link:** https://library.si.edu/digital-library/book/meychaniqueanal00lagr  
**Reference pages:** Smithsonian Libraries record, Internet Archive scan, Craig Fraser's preface to the English translation, Knill and Luenberger on the genesis of Lagrange multipliers

## The Problem That Made This Necessary

Euler made optimization big enough to handle curves. That solved one problem and exposed another:

```text
Real systems are not free to vary arbitrarily.
```

Curves have fixed endpoints. Mechanical systems have joints, rods, surfaces, conservation laws, and geometric restrictions. Statistical models have normalization constraints. Modern neural nets have regularizers, resource limits, and parameter-sharing structure.

If ordinary calculus says "set the derivative to zero," constrained problems ask a harder question:

```text
What does "zero change" mean when only some changes are allowed?
```

Eliminating the constraints by hand is sometimes possible, but it quickly becomes ugly. Worse, elimination can hide the geometry of the problem. Lagrange's move was to keep the constraints visible and introduce extra unknowns that measure the force or pressure those constraints exert.

That is the pressure that made Lagrange's analytical mechanics and multiplier method necessary. Optimization needed a language for freedom under conditions.

## Key Ideas

### 1. Mechanics becomes analysis

Lagrange's famous promise was that mechanics could be done analytically, without leaning on geometric diagrams for each case. In the Smithsonian summary, the book is described as replacing older synthetic treatments of mechanics with an analytical system.

That is not just a stylistic preference. It changes what counts as a method.

Newton's mechanics often solved problems by drawing the geometry of a particular system. Lagrange recast mechanics into equations built from coordinates, variations, constraints, and energies. Once the problem is written analytically, the same machinery can be reused across pendulums, particles, rigid bodies, fluids, and later fields.

Modern ML inherits this habit deeply:

```text
represent the system -> write the objective -> write the constraints -> optimize
```

### 2. Virtual displacements isolate allowed change

The early part of *Méchanique analitique* develops the principle of virtual velocities or virtual work. The core idea is that one should consider infinitesimal possible displacements consistent with the constraints of the system.

This is the constrained version of the local-change idea from Fermat and the variational idea from Euler:

```text
not every tiny change is legal
```

For an unconstrained variable, a local optimum must resist every small perturbation. For a constrained system, the relevant perturbations are only the allowed ones. Lagrange gives that restriction an analytic grammar.

This matters because constrained optimization is really about tangent directions. At a constrained optimum, the objective does not need to be flat in every direction. It needs to be flat along the feasible surface.

### 3. Multipliers keep constraints inside the calculation

The multiplier method is the durable mathematical artifact.

In modern notation, instead of optimizing

```text
f(x) subject to g(x) = 0
```

by eliminating `g`, we form

```text
L(x, lambda) = f(x) - lambda g(x)
```

and solve stationarity conditions involving both the original variables and the new multiplier `lambda`.

The multiplier is not a trick decoration. It says that, at the optimum, the gradient of the objective is balanced by the gradient of the constraint:

```text
objective pressure = constraint pressure
```

In mechanics, that pressure can correspond to constraint forces. In optimization, it becomes shadow price, dual variable, penalty weight, or sensitivity.

That is why this note belongs after Euler. Euler widened optimization to functions and curves; Lagrange gave constrained variation a reusable analytic form.

### 4. The Lagrangian packages objectives and constraints

The word "Lagrangian" now appears in mechanics, optimization, statistics, and machine learning. The contexts differ, but the shape is familiar:

```text
main quantity to optimize + terms that encode the structure of the problem
```

In mechanics, this leads toward the difference between kinetic and potential energy and the Euler-Lagrange equations. In optimization, it leads toward constrained extrema, duality, KKT conditions, augmented Lagrangians, and regularized objectives.

The line from Lagrange to modern ML is not decorative. Every time we write a constrained or penalized objective, we are doing mathematics in this style:

```text
make the tradeoff explicit, then optimize the combined analytic object
```

## Historical Context

Lagrange was working after Newton, Leibniz, the Bernoullis, d'Alembert, and Euler. By 1788, calculus and the calculus of variations had already shown their power, but mechanics was still scattered across geometric arguments, special cases, and physical intuition.

*Méchanique analitique* tried to consolidate that world. Craig Fraser notes that it appeared almost exactly a century after Newton's *Principia* and represented the culmination of a program to recast mechanics in the analytic style of the Leibnizian calculus. The book is mechanics after the algebraic and variational revolutions have matured.

The multiplier method also has a historical ambiguity worth keeping honest. Lagrange did not write modern vector notation, modern gradients, or KKT conditions. The method was embedded in mechanics, statics, and variations. The clean textbook form came later.

But the lineage point is solid: Lagrange made constraints first-class analytic objects. He did not merely solve a constrained problem; he created a method where the constraint remains in the equation and receives its own unknown.

## Lineage

```text
Fermat (local extrema by perturbation)
  └── Newton & Leibniz (calculus: rates and accumulation)
      └── Euler 1744 (optimize curves/functions)
          └── Lagrange 1788 (constrained variation, multipliers, analytical mechanics)
              ├── Hamilton/Jacobi mechanics (action principles)
              ├── constrained optimization and dual variables
              ├── Cauchy 1847 (descent methods for finite variables)
              └── modern ML objectives (regularization, constrained training, variational inference)
```

**Built on:** [[Euler - Calculus of Variations 1744|Euler]], because optimization had to widen from numbers to curves/functions before constrained variation could become a general method; [[../05-Newton-Leibniz/Newton-Leibniz - Fundamental Theorem of Calculus|Newton and Leibniz]], because analytical mechanics depends on differential and integral calculus; and [[../04-Early-Calculus/Fermat - Adequality and Tangents 1637|Fermat]], because local extrema begin as perturbation arguments.  
**Enables:** Hamiltonian and Lagrangian mechanics, constrained optimization, multiplier methods, duality, optimal control, variational inference, and regularized machine-learning objectives.  
**Later echoes:** KKT conditions, Lagrangian relaxation, augmented Lagrangian methods, constrained maximum likelihood, entropy maximization, support vector machines, and resource-aware neural network training.

## Why It Matters for Mathematics

Lagrange changes optimization from "find the best object" to:

```text
find the best object under the structure the problem imposes
```

That is the real world. Free optimization is the clean classroom case. Constraints are where mathematics meets bodies, data, budgets, probabilities, and physics.

The multiplier method also introduces one of the most powerful ideas in applied mathematics: extra variables can reveal structure rather than add mess. The multiplier is not part of the original object being optimized, but adding it makes the hidden balance of the problem visible.

That pattern will return again and again:

```text
latent variables, dual variables, slack variables, hidden states
```

Good mathematics sometimes solves a hard problem by expanding the space in which the problem is expressed.

## Why It Matters for AI

AI systems are trained by optimization, but practical AI is rarely unconstrained optimization.

Probabilities must sum to one. Losses are regularized. Models are trained under compute budgets. Reinforcement learning policies optimize accumulated reward under dynamics. Variational inference optimizes an evidence lower bound over distributions. Constrained decoding, calibration, fairness constraints, energy limits, and safety filters all ask the same old question:

```text
what is the best solution that remains feasible?
```

Lagrange is one of the deep ancestors of that question.

The AI bridge is especially clear in three places:

- **Regularization:** add a penalty term to the objective, often interpretable as a relaxed constraint.
- **Variational inference:** optimize over distributions while respecting normalization and divergence constraints.
- **Dual methods:** turn hard constrained problems into objective-plus-multiplier problems that can be optimized iteratively.

Gradient descent tells a model how to move downhill. Lagrange tells us what "downhill" means when the path is fenced by constraints.

## Why This Note Belongs Here

The Mathematics lineage already had:

```text
Euler 1744 -> Cauchy 1847 -> Robbins-Monro -> SGD -> Adam
```

That path explains descent, but it skips the constraint language used everywhere in modern optimization. Lagrange closes that gap.

Without Lagrange, the chain makes optimization look too free. With him, the lineage becomes more honest:

```text
optimize a function, but keep the problem's structure alive
```

That is modern AI in one sentence.

## References

- Lagrange, J.-L. *Méchanique analitique*. 1788. `corpus/raw/library-papers/lagrange-analytical-mechanics-1788.pdf`
- Smithsonian Libraries and Archives. "Méchanique analitique." https://library.si.edu/digital-library/book/meychaniqueanal00lagr
- Internet Archive file record. https://archive.org/details/meychaniqueanal00lagr
- Fraser, C. G. "Preface to J. L. Lagrange, Analytical Mechanics." https://cfraser.artsci.utoronto.ca/Lagrange1997preface.pdf
- Knill, O. and Luenberger, D. G. "On the Genesis of the Lagrange Multipliers." https://legacy-www.math.harvard.edu/~knill/teaching/summer2014/exhibits/lagrange/genesis_lagrangemultpliers.pdf

## Status

- [x] Primary source downloaded to `corpus/raw/library-papers/lagrange-analytical-mechanics-1788.pdf`
- [x] PDF verified with `file` as a PDF document
- [x] Note created as the bridge from Euler's variational optimization to constrained optimization
- [ ] Future note: Hamilton/Jacobi mechanics and action principles
- [x] Follow-up note completed: [[Cauchy - Steepest Descent 1847|Cauchy 1847 steepest descent]]

---

*Euler asks which path is best. Lagrange asks which path is best when the world says, "yes, but under these conditions." That "yes, but" is where applied mathematics lives.*
