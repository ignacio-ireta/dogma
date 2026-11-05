# Pierre de Fermat — Adequality, Maxima/Minima, and Tangents (1637)

#person #primary-source #milestone #early-calculus #optimization #lineage

**Person:** Pierre de Fermat (1607–1665)  
**Work:** *Methodus ad disquirendam maximam et minimam et de tangentibus linearum curvarum* — “Method for investigating maxima and minima and tangents to curved lines”  
**Date:** circulated around 1637; printed posthumously in Fermat’s collected works  
**Primary source:** Fermat, in *Œuvres de Fermat*, ed. Paul Tannery & Charles Henry, with the relevant maxima/minima and tangent texts collected around vol. I pp. 133–172 / supplement pp. 120ff.  
**Paper:** `corpus/raw/library-papers/fermat-maxima-minima-1637.pdf`  
**Secondary source:** `corpus/raw/library-papers/stromholm-fermat-1965.pdf` — Per Strømholm, “Fermat’s Methods of Maxima and Minima and of Tangents. A Reconstruction.”  
**Source link:** https://archive.org/details/oeuvresdefermatp00ferm

## The Problem That Made This Necessary

Descartes made curves algebraic. Once a curve could be written as an equation, a new problem immediately appeared:

```text
How do we extract local behavior from the equation?
```

Ancient geometry could draw a tangent in special cases. Algebra could solve equations. But analytic geometry needed something sharper: a procedure for finding where a curve turns, where a quantity is largest or smallest, and what line just touches a curve at a point.

That is the missing bridge between coordinates and calculus.

Fermat’s answer was **adequality**: compare a quantity with its value after a tiny algebraic perturbation, cancel what is common, divide away the perturbation, then ignore the remaining higher-order scraps. He did not have limits, derivatives, or epsilon-delta rigor. But he had the operational skeleton of differentiation.

## Key Ideas

### 1. Extremes reveal themselves by almost-equality

For a quantity depending on an unknown `A`, Fermat introduced a small increment, often written `E`, and compared the expression at `A` with the expression at `A + E`.

At a maximum or minimum, the two nearby values are not exactly equal, but they are “adequal” — close enough that the first-order behavior carries the decisive information.

In modernized form, the pattern looks like this:

```text
f(A + E) adequals f(A)
expand
cancel common terms
divide by E
suppress remaining terms containing E
solve the resulting equation
```

That last equation is what we would now recognize as `f′(A) = 0`.

This is why Fermat belongs before Newton and Leibniz. The full derivative concept is not here yet, but the extremum test is already trying to be born.

### 2. Tangents become algebraic, not merely geometric

Fermat used related reasoning for tangents. A tangent line is locally almost the same as the curve near the point of contact. If analytic geometry turns the curve into an equation, then the tangent problem becomes algebraic: compare the curve and a nearby line, use the small displacement, and solve for the line that fits the local behavior.

This matters because the tangent is the geometric face of the derivative.

Before calculus had a formal language, the tangent problem was one of the pressures forcing that language into existence.

### 3. This is not yet modern calculus — and that is exactly why it matters

Strømholm’s reconstruction is useful because it warns against making Fermat too modern. Fermat was not simply “taking a derivative” in our notation. Historians have disagreed about whether his method should be read through infinitesimals, double roots, or other later concepts.

The safer reading is lineage-based:

- Fermat had an algorithm for local algebraic comparison.
- The algorithm worked for maxima, minima, and tangents.
- Later calculus supplied the conceptual machinery that explained why it worked.

That is often how mathematics grows: a technique becomes reliable before its foundations become clean.

## Historical Context

Fermat and Descartes were mathematical rivals. Descartes’ *La Géométrie* appeared in 1637 and helped define analytic geometry. Fermat had developed related coordinate methods independently and earlier, but much of his work circulated privately through letters and manuscripts rather than as a polished book.

This matters. Fermat’s ideas entered the mathematical bloodstream unevenly. His method of maxima and minima was known, debated, criticized, and reinterpreted before the cleaned-up historical record existed. Strømholm notes that the confusion began already in Fermat’s own lifetime because essays and letters circulated in manuscript selections.

So Fermat’s role is not “the inventor of calculus” in the simple schoolbook sense. His role is subtler and more interesting: he exposed the exact local problems that analytic geometry could state but not yet systematically solve.

## Lineage

```text
Viète 1591 (symbolic algebra: letters make general forms visible)
  └── Descartes 1637 (analytic geometry: equations become curves)
      └── Fermat 1637 (adequality: local comparison, extrema, tangents)
          ├── Newton & Leibniz (calculus: derivative/integral machinery)
          ├── Euler (functions and series become the main medium)
          ├── Cauchy → Weierstrass (limits and rigor)
          └── Cauchy 1847 → gradient descent → modern optimization
```

**Built on:** [[../02-Algebraic-Revolution/Viète - Symbolic Notation 1591|Viète]] and [[../03-Analytic-Geometry/Descartes - Analytic Geometry 1637|Descartes]]. Algebra had to become symbolic, and curves had to become equations, before the local behavior of curves could become a general algebraic problem.  
**Enables:** [[../05-Newton-Leibniz/Newton-Leibniz - Fundamental Theorem of Calculus|Newton and Leibniz]], because differentiation begins as the problem of tangents and extrema before it becomes a formal operator.  
**Later echoes:** optimization theory, critical points, gradients, backpropagation, loss minimization.

## Why It Matters for Mathematics

Fermat changes the question from global shape to local change.

Archimedes and exhaustion asked: *How much area is there?*  
Descartes asked: *What equation describes this curve?*  
Fermat asks: *What happens if I nudge the input slightly?*

That nudge is enormous.

Once mathematics starts asking about a small perturbation, it can discover slopes, turning points, stability, approximation, and eventually differential equations. Fermat’s adequality is not rigorous by later standards, but it is pointed in exactly the right direction: toward a mathematics of local variation.

## Why It Matters for AI

Training a model is mostly the industrial-scale version of Fermat’s question:

```text
If I nudge this parameter, does the objective improve or worsen?
```

Modern machine learning wraps that question in vector calculus, automatic differentiation, stochastic estimates, GPUs, and enormous parameter spaces. But the conceptual ancestor is still visible:

- define a quantity to optimize,
- perturb a variable,
- compare the change,
- find where improvement stops or changes direction.

A loss minimum in a neural network is not historically separate from Fermat’s maxima/minima problem. It is the same family of thought, scaled from one algebraic variable to billions of weights.

That is the thread from early calculus to AI: **local change becomes optimization**.

## Why This Note Belongs Here

The Mathematics lineage had a gap:

```text
Descartes → Newton/Leibniz
```

That jump was too clean. Calculus did not appear immediately after coordinates. Between them came the concrete pressure of tangent lines and extrema — problems that analytic geometry made newly algebraic but did not fully solve.

Fermat supplies that missing hinge:

```text
symbolic algebra → coordinate curves → local perturbation → derivative → optimization
```

Now the chain toward modern AI is more honest.

## References

- Fermat, P. de. *Œuvres de Fermat*, ed. P. Tannery & C. Henry. Internet Archive scan. `corpus/raw/library-papers/fermat-maxima-minima-1637.pdf`
- Strømholm, P. “Fermat’s Methods of Maxima and Minima and of Tangents. A Reconstruction.” `corpus/raw/library-papers/stromholm-fermat-1965.pdf`
- Katz, M. G., Schaps, D. M., & Shnider, S. “Almost Equal: The Method of Adequality from Diophantus to Fermat and Beyond.” arXiv:1210.7750. `corpus/raw/library-papers/1210.7750.pdf` — used only for orientation on the later historiography of adequality.

## Status

- [x] Primary source downloaded to `corpus/raw/library-papers/fermat-maxima-minima-1637.pdf`
- [x] Secondary reconstruction downloaded to `corpus/raw/library-papers/stromholm-fermat-1965.pdf`
- [x] Modern historiography paper downloaded to `corpus/raw/library-papers/1210.7750.pdf`
- [x] PDFs verified with `file` as PDF documents
- [x] Note created as the bridge from analytic geometry to Newton-Leibniz calculus
- [ ] Future note: Cavalieri’s indivisibles — the parallel integration-side pressure toward calculus

---

*Fermat did not yet have the derivative. He had something more historically revealing: the itch that made the derivative necessary.*
