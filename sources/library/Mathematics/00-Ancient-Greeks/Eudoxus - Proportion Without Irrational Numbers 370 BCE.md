# Eudoxus - Proportion Without Irrational Numbers (~370 BCE)

#person #primary-source #milestone #proportion #incommensurability #lineage

**Person:** Eudoxus of Cnidus (~408-355 BCE)  
**Work preserved through:** Euclid, *Elements*, Book V  
**Date:** Eudoxus's theory ~370 BCE; Euclid's compilation ~300 BCE  
**Primary source:** *Euclid's Elements in Greek, Volume II: Books 5-9*, modern English translation based on Heiberg's Greek text  
**Paper:** `corpus/raw/library-papers/eudoxus-proportion-book-v-300bce.pdf`  
**Source link:** https://math.berkeley.edu/~wodzicki/160.F05/vol2.pdf  
**Source note:** No text written by Eudoxus survives. Book V of Euclid's *Elements* is the earliest extant presentation of the general theory of proportion traditionally attributed to him. The local PDF was downloaded and verified with `file` as a PDF document.

## Paper/Source Reference

Euclid. *Elements*, Book V, in *Euclid's Elements in Greek, Volume II: Books 5-9*, based on J. L. Heiberg's Greek text, with modern English translation and notes. Local PDF: `corpus/raw/library-papers/eudoxus-proportion-book-v-300bce.pdf`.

The source's introduction identifies Book V's theory of proportion as generally attributed to Eudoxus. That qualification matters. We possess Euclid's preservation of the theory, not an autograph or surviving treatise by Eudoxus. The lineage should therefore distinguish historical attribution from textual survival.

## The Problem That Made This Necessary

Greek geometry discovered magnitudes that could not be measured by one common unit. The diagonal of a unit square has length √2: no whole-number multiple of the side equals a whole-number multiple of the diagonal. The two lengths are **incommensurable**.

That broke the easy arithmetic picture of ratio.

If a ratio were merely a fraction of whole numbers, then some geometrically legitimate ratios would not exist. Yet geometers still needed to compare lengths, areas, and volumes and to prove that two such comparisons were equal. They could not simply write decimal expansions, invoke real numbers, or treat √2 as a completed numerical object; none of that machinery existed.

The problem was therefore sharper than "how do we calculate a ratio?"

```text
How can we compare arbitrary magnitudes rigorously
when no common measuring unit and no rational number represents their ratio?
```

Eudoxus's answer was to define equality of ratios through the behavior of **all integer multiples**. He did not need to say what the ratio itself was as a number. He specified what it means for two ratios to be the same.

This is the missing hinge between Zeno and Euclid. Zeno shows that continuity and infinite division make naive reasoning unreliable. Eudoxus supplies a finite, quantifier-driven comparison method that works even for incommensurable magnitudes. Euclid then preserves that method inside a deductive system, and Archimedes uses the same style of controlled comparison in exhaustion arguments.

## Key Ideas

### 1. Ratio is defined by comparisons, not by a hidden number

Book V's central definition says, in modernized language, that magnitudes `a` and `b` have the same ratio as `c` and `d` when, for every pair of positive integers `m` and `n`, the comparisons agree:

```text
ma > nb  exactly when  mc > nd
ma = nb  exactly when  mc = nd
ma < nb  exactly when  mc < nd
```

This is extraordinarily careful. It never requires division of magnitudes. It never assumes that `a/b` is a rational number. It tests the two ratios by every possible pair of integer multiples and declares them equal when they have the same ordering pattern.

The ratio becomes known by its relations.

### 2. Incommensurable magnitudes become mathematically usable

Suppose one pair of magnitudes shares a common unit and another does not. Earlier arithmetic treatments could handle the first pair but risked excluding the second. Eudoxus's definition applies to both.

The trick is not to force every magnitude into the integers. It is to let integers probe the magnitudes:

```text
integers do not constitute the continuum;
their multiples test relationships within it.
```

That lets Greek geometry reason uniformly about rational and irrational-looking ratios centuries before a rigorous construction of the real numbers.

### 3. The Archimedean condition rules out unreachable magnitudes

Book V also requires magnitudes that have a ratio to be capable, when multiplied, of exceeding one another. Given magnitudes of the same kind, enough copies of the smaller must eventually surpass the larger.

In modern language this is related to the Archimedean property. It excludes infinitesimal or infinitely large magnitudes that remain forever beyond finite multiples.

This condition makes the comparison scheme work. If finite multiples could never cross, the integer tests would fail to locate one magnitude relative to another. Eudoxian proportion therefore controls continuity without admitting actual infinitesimals.

### 4. The theory is an ancestor of epsilon-style rigor

Eudoxus did not invent limits, real numbers, or epsilon-delta analysis. Reading those concepts directly into Book V would flatten the history.

But the family resemblance is real. A statement about continuous magnitude is made rigorous through a rule that must survive **every** choice of integers. Intuition is replaced by quantified comparison.

```text
for every m and n, the order relations agree
```

That is an early example of a pattern modern analysis will make explicit:

```text
replace a vague picture of continuity with a condition over all admissible tests
```

Dedekind later constructs real numbers using cuts in the rationals; Weierstrass defines limits through quantified bounds. Eudoxus does neither, but he demonstrates the deeper move: characterize an elusive continuous object by the comparisons it must satisfy.

### 5. Proportion becomes reusable proof infrastructure

Book V does not merely offer one clever definition. It proves an algebra of proportions: ratios can be transformed, alternated, compounded, and compared while preserving rigor.

That infrastructure is essential for the later books of the *Elements* and for Greek geometry more broadly. Similar figures, areas, and volumes depend on proportion. Archimedes' exhaustion arguments depend on comparing magnitudes tightly enough that a supposed excess or deficit leads to contradiction.

Eudoxus makes continuous magnitude governable before it becomes numerical in the modern sense.

## Historical Context

The discovery of incommensurable magnitudes is traditionally associated with the Pythagorean world. Whatever the exact story, Greek mathematicians faced a genuine foundational crisis: geometry produced relationships that whole-number arithmetic could not express.

Eudoxus, a mathematician and astronomer associated with Cnidus and Plato's Academy, is traditionally credited with two connected achievements:

- the general theory of proportion preserved in Euclid's Book V;
- the method of exhaustion later used with exceptional power by Archimedes.

Neither survives in a text under Eudoxus's own name. Our evidence is mediated by later authors. Euclid preserves the proportion theory without naming its inventor; Archimedes explicitly credits Eudoxus with proofs concerning pyramids, prisms, cones, and cylinders. This note uses Book V as the primary mathematical source and keeps the attribution appropriately cautious.

The conceptual unity is strong. Proportion handles arbitrary magnitudes by exhaustive integer comparison. Exhaustion handles areas and volumes by finite approximations whose remaining difference can be made smaller than any assigned magnitude. Both methods avoid casual appeals to completed infinity while still reasoning about the continuum.

## Lineage

### What came before

```text
Pythagorean arithmetic
  -> ratio is naturally understood through whole numbers
Discovery of incommensurability
  -> some geometric magnitudes have no common unit
Zeno ~450 BCE
  -> infinite division exposes the danger of naive continuity arguments
Eudoxus ~370 BCE
  -> arbitrary magnitudes are compared through all integer multiples
```

**Built on:** [[Zeno Paradoxes|Zeno's paradoxes]] as the pressure for disciplined reasoning about continuity, and the Greek discovery of incommensurable magnitude as the specific failure of fraction-based ratio.

### What Eudoxus enabled

```text
Eudoxian proportion
  -> Euclid Book V: a general theory of ratio
  -> Euclid's geometry can include incommensurable magnitudes
  -> exhaustion compares curved magnitudes without actual infinity
  -> Archimedes proves areas and volumes rigorously
  -> Cavalieri makes slicing operational
  -> Newton and Leibniz turn accumulation into calculus
  -> Weierstrass and Dedekind rebuild continuity with modern rigor
```

**Enables:** [[Euclid - Elements and Axiomatic Geometry 300 BCE|Euclid]], who preserves the theory inside an axiomatic proof system; [[Archimedes - Method of Exhaustion|Archimedes]], who turns Eudoxian comparison and exhaustion into proto-integration; [[../08-Set-Theory-Infinity/Dedekind - Real Numbers and Cuts 1872|Dedekind]], whose real-number construction solves in arithmetic form a continuum problem the Greeks had managed geometrically.

The local chain is now explicit:

```text
incommensurability -> comparison without division -> axiomatic proof
-> exhaustion -> integration -> rigorous real analysis
```

## Why It Matters for AI

Modern AI rarely invokes Eudoxian ratios directly. The inheritance is methodological and foundational.

AI depends on continuous quantities: real-valued parameters, distances, probabilities, gradients, losses, and convergence tolerances. Those quantities only become trustworthy because mathematics learned to separate an object from the tests that characterize it.

Eudoxus gives an early, powerful version of that habit:

```text
do not demand a convenient representation;
define equivalence by observable comparison behavior.
```

That idea echoes through modern mathematics and computation:

- real numbers can be characterized by rational bounds;
- limits are characterized by their response to every tolerance;
- probability distributions can be compared through expectations or divergences;
- vectors and models are often known through their action on inputs;
- convergence is not "it seems close" but a quantified guarantee.

There is also a direct numerical lesson. Machine learning works with approximations everywhere. Floating-point numbers, sampled expectations, iterative optimization, and finite datasets stand in for ideal continuous objects. The central question is the Eudoxian one: **what comparisons or error conditions make the approximation legitimate?**

Eudoxus matters because he shows mathematics learning to reason reliably about quantities it cannot express as ordinary numbers. That is one of the oldest roots of the rigor beneath continuous optimization and numerical AI.

## References

- Euclid. *Elements*, Book V. Local primary-source PDF: `corpus/raw/library-papers/eudoxus-proportion-book-v-300bce.pdf`
- Direct PDF source used for download: https://math.berkeley.edu/~wodzicki/160.F05/vol2.pdf
- The source introduction, Book V definitions 4-5, and the propositions developing equality and transformation of ratios.
- Archimedes. *The Method of Archimedes*, for the ancient attribution of volume theorems to Eudoxus and the later use of exhaustion. `corpus/raw/library-papers/archimedes-method-250bce.pdf`

## Status

- [x] Extant primary text of the attributed theory downloaded to `corpus/raw/library-papers/eudoxus-proportion-book-v-300bce.pdf`
- [x] PDF verified with `file` as a PDF document
- [x] Absence of any surviving Eudoxus-authored text documented
- [x] Note created as the incommensurability-to-rigor hinge between Zeno and Euclid
- [x] Cross-linked through the Mathematics MOC and LINEAGE map

---

*Eudoxus did not make the irrational rational. He made it possible to reason without pretending the problem was smaller than it was.*
