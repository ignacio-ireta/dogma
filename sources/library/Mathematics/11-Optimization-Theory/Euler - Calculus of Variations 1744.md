# Leonhard Euler — Calculus of Variations (1744)

#person #primary-source #milestone #optimization #calculus-of-variations #lineage

**Person:** Leonhard Euler (1707-1783)  
**Work:** *Methodus inveniendi lineas curvas maximi minimive proprietate gaudentes, sive solutio problematis isoperimetrici latissimo sensu accepti*  
**Date:** 1744  
**Primary source:** Euler's 1744 treatise on curves with maximum or minimum properties, digitized from the public-domain text  
**Paper:** `corpus/raw/library-papers/euler-calculus-of-variations-1744.pdf`  
**Source link:** https://www.wilbourhall.org/pdfs/euler/Methodus_inveniendi_lineas_curvas.pdf  
**Reference pages:** Euler Archive / University of the Pacific entry, Internet Archive scan, Craig Fraser's historical chapter on Euler's calculus of variations

## The Problem That Made This Necessary

Fermat could optimize a quantity. Newton and Leibniz could differentiate functions and relate rates to accumulated areas. But a new class of problems refused to fit inside ordinary calculus:

```text
What if the unknown is not a number, but an entire curve?
```

The brachistochrone problem asked for the path of fastest descent. The isoperimetric problem asked which closed curve encloses the greatest area for a fixed perimeter. Mechanics asked which shape or path nature chooses under constraints.

These were not just questions about finding `x` where `f(x)` is largest. They asked for the best object among infinitely many possible functions or curves.

That is the pressure that made the calculus of variations necessary. Ordinary calculus optimizes points. Euler begins the systematic optimization of functions.

## Key Ideas

### 1. A curve can be the unknown

Euler's treatise made a decisive conceptual shift. Instead of treating a curve as something already given by an equation, he treated the curve itself as the thing to be found.

That sounds small. It is not.

It means mathematics can ask:

```text
Among all possible curves satisfying these constraints, which one optimizes a quantity?
```

This is the birth of the functional viewpoint: the input is a function or curve, and the output is a number such as length, time, area, energy, or action.

Modern notation would write many of these problems as optimizing an integral:

```text
J[y] = integral F(x, y, y') dx
```

Euler did not write it in this later notation, but the lineage is already there. The target of optimization has moved one level up.

### 2. Optimization becomes a differential equation

Euler's central move was to derive necessary conditions that an optimizing curve must satisfy. Instead of guessing curves one by one, he showed how variational problems could produce differential equations.

In later language, this becomes the Euler equation, and after Lagrange's reformulation, the Euler-Lagrange equation:

```text
d/dx (partial F / partial y') - partial F / partial y = 0
```

The exact notation belongs to later history. The important lineage point is simpler:

```text
best curve -> necessary local condition -> differential equation
```

That move is enormous. It turns optimization over an infinite family of possible shapes into an analytic condition that can be solved, studied, and generalized.

### 3. Constraints become part of optimization theory

Euler's title points directly at the isoperimetric problem: maximize or minimize a quantity while another quantity is held fixed. That is constrained optimization before the modern vocabulary exists.

This matters because constraints are not decorative. They change the problem. The best free curve and the best curve under a length, area, boundary, or physical constraint are different objects.

The later Lagrange multiplier method grows from this soil. Modern constrained optimization, regularized loss functions, and physics-informed objectives all inherit the same pattern:

```text
optimize something, but under conditions that cannot be ignored
```

## Historical Context

The calculus of variations did not appear from nowhere. It grew out of the late seventeenth-century challenge problems of the Bernoulli family, especially the brachistochrone and isoperimetric problems. Those problems exposed a weakness in ordinary calculus: it could handle local extrema of a known function, but not systematic extrema over families of curves.

Euler's 1744 *Methodus inveniendi* is usually treated as the work that established the calculus of variations as a distinct field. The University of the Pacific's Euler Archive summary describes the book as moving the subject from special cases to general classes of problems, and Craig Fraser's historical account emphasizes that Euler organized a broad theory for optimizing integral quantities.

There is an important historical wrinkle: Lagrange later introduced a different and more powerful variational notation, and the subject is now often remembered through the combined phrase "Euler-Lagrange." But Euler's 1744 book is the necessary node here. It is where the problem becomes general enough to be a field rather than a bag of clever solutions.

## Lineage

```text
Archimedes (areas by exhaustion)
  └── Cavalieri (figures compared by slices)
      └── Newton & Leibniz (calculus: rates and accumulation)
          ├── Fermat (local extrema of quantities)
          └── Euler 1744 (extrema of curves/functions)
              ├── Lagrange (variational notation, multipliers)
              ├── Hamilton/Jacobi mechanics (action principles)
              ├── Cauchy 1847 (steepest descent for finite variables)
              └── modern optimization (loss landscapes, constraints, function spaces)
```

**Built on:** [[../04-Early-Calculus/Fermat - Adequality and Tangents 1637|Fermat]], because maxima/minima had to become algebraic before optimization could be generalized; [[../05-Newton-Leibniz/Newton-Leibniz - Fundamental Theorem of Calculus|Newton and Leibniz]], because variational problems depend on differentiating and integrating changing quantities; and [[../04-Early-Calculus/Cavalieri - Indivisibles 1635|Cavalieri]], because the objective is often an accumulated quantity over a continuous curve.  
**Enables:** Lagrange's variational calculus and multiplier methods, Hamilton's principle of least action, Cauchy's descent method, and the modern idea that learning is optimization over a high-dimensional function family.  
**Later echoes:** constrained optimization, regularization, optimal control, variational inference, energy-based models, and neural network training.

## Why It Matters for Mathematics

Euler changes the scale of optimization.

Fermat asks:

```text
Which value of x makes this expression largest or smallest?
```

Euler asks:

```text
Which entire curve makes this accumulated quantity largest or smallest?
```

That move opens a new kind of mathematical object: the functional. Once functionals exist, mathematics can reason about spaces of functions, not just numbers and finite-dimensional coordinates. This is one of the routes from classical calculus to functional analysis, modern mechanics, control theory, and optimization.

The calculus of variations also makes a philosophical claim that later science keeps reusing: global behavior may be characterized by an extremal principle. Nature may choose a path by minimizing time, action, energy, or another quantity. Even when that language is only a model, it gives mathematics a powerful way to derive governing equations.

## Why It Matters for AI

Machine learning is optimization, but not only in the small Fermat sense.

A model is a parameterized function. Training asks for the member of a function family that minimizes a loss. Regularization adds constraints or penalties. Variational inference turns probabilistic approximation into an optimization problem over distributions. Optimal control and reinforcement learning ask for policies that optimize accumulated reward over trajectories.

Euler's 1744 move is therefore one of the deep ancestors of modern AI:

```text
optimize over curves/functions -> optimize over model families
```

Neural networks are finite-dimensional in their parameters, so Cauchy's steepest descent is the more direct algorithmic ancestor. But the larger conceptual ancestor is Euler: optimization is not just finding a number. It is selecting the best function-like object under a criterion.

That is why this note belongs in the chain. Without calculus of variations, the lineage jumps from early extrema to gradient descent too abruptly. Euler supplies the missing widening step: from local change to optimization over spaces.

## Why This Note Belongs Here

The Mathematics lineage had a clear optimization gap:

```text
Fermat / Newton-Leibniz -> Cauchy steepest descent -> modern ML optimization
```

That path is true, but too narrow. Before Cauchy could turn gradients into an iterative algorithm, mathematics had to learn that optimization could be a general organizing principle for curves, functions, constraints, and physical systems.

Euler's *Methodus inveniendi* is that hinge.

## References

- Euler, L. *Methodus inveniendi lineas curvas maximi minimive proprietate gaudentes*. 1744. `corpus/raw/library-papers/euler-calculus-of-variations-1744.pdf`
- Euler Archive / University of the Pacific. "Methodus inveniendi lineas curvas maximi minimive proprietate gaudentes." https://scholarlycommons.pacific.edu/euler-works/65/
- Internet Archive scan record. https://archive.org/details/methodusinvenie00eule
- Fraser, C. G. "Leonhard Euler, Book on the Calculus of Variations (1744)." `https://cfraser.artsci.utoronto.ca/LandmarksEuler.pdf`

## Status

- [x] Primary source downloaded to `corpus/raw/library-papers/euler-calculus-of-variations-1744.pdf`
- [x] PDF verified with `file` as a PDF document
- [x] Note created as the bridge from calculus to optimization theory
- [x] Follow-up note completed: [[Lagrange - Analytical Mechanics and Multipliers 1788|Lagrange's variational method and multipliers]]
- [x] Follow-up note completed: [[Cauchy - Steepest Descent 1847|Cauchy 1847 steepest descent]]

---

*Euler widened the question. Not "which number is best?" but "which possible world, path, or function is best?" Modern AI still lives inside that question.*
