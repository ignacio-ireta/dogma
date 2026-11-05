# George Boole — Algebra of Logic (1854)

#person #book #milestone #mathematical-logic #boolean-algebra #ai-lineage

**Person:** George Boole (1815–1864)  
**Work:** *An Investigation of the Laws of Thought, on Which Are Founded the Mathematical Theories of Logic and Probabilities* (1854)  
**Primary source:** George Boole, Walton and Maberly, London, 1854  
**Paper:** `corpus/raw/library-papers/boole-laws-of-thought-1854.pdf`  
**Source note:** The local file is Project Gutenberg ebook 15114, a complete 343-page PDF transcription of the 1854 book. It was downloaded from Project Gutenberg and verified with `file` as a PDF document, version 1.5.

## Title
Reasoning becomes an algebra that can be calculated.

Traditional logic could classify syllogisms and judge familiar argument forms. But a catalogue of verbal forms did not provide a general procedure for reasoning through many interacting classes, exclusions, and conditions. Algebra, meanwhile, already showed how a small stock of symbols and transformation laws could solve whole families of quantitative problems.

Boole's question was the necessary one:

> Can the operations of reasoning themselves be represented by symbols, governed by fixed laws, and solved by calculation?

His answer was an algebra of classes. A symbol denotes a class of things; multiplication selects their intersection; subtraction excludes a subclass; `1` denotes the universe; and `0` the empty class. Its decisive law is idempotence, $x^2=x$: selecting the things that are $x$ and then selecting $x$ again changes nothing. Because the only ordinary numbers satisfying that equation are 0 and 1, logical calculation acquires a binary algebraic domain.

This does not yet give modern Boolean algebra, digital circuitry, or neural networks. It gives the missing mathematical permission: truth conditions and class membership can be encoded as discrete symbols and transformed mechanically. Later logic regularizes Boole's system; switching theory gives it a physical implementation; McCulloch and Pitts make firing neurons realize logical expressions.

## Paper / Source Reference

**Boole, G. (1854).** *An Investigation of the Laws of Thought, on Which Are Founded the Mathematical Theories of Logic and Probabilities.* London: Walton and Maberly.

Local source:
- `corpus/raw/library-papers/boole-laws-of-thought-1854.pdf`

Download source used:
- `https://www.gutenberg.org/files/15114/15114-pdf.pdf`

Primary passages: Chapter I (the program of a symbolic calculus), Chapter II (signs, class multiplication, and $x^2=x$), Chapter III (the restriction to 0 and 1), and Chapter V (symbolic reasoning and expansion).

## What Problem Made This Necessary?

Aristotelian logic organized valid forms such as “All $A$ are $B$” and “No $A$ are $B$.” That was useful but local: families of syllogisms were named and checked in verbal form. It offered no compact general algorithm for eliminating intermediate terms or deriving consequences from complicated premises.

Symbolic algebra had solved the analogous problem for quantity. Once a problem became equations, one could transform the representation while preserving what it meant. Boole's opening program is explicitly to express the operations of reasoning “in the symbolical language of a Calculus” and use that calculus to construct a general method (Chapter I, p. 1).

The needed step was therefore a change of representation:

```text
verbal argument
  -> symbolic constraints
  -> law-preserving algebraic transformations
  -> interpretable conclusion
```

That pattern is an ancestor of both computer programs and machine-learning models: encode a problem into mathematical state, operate on the encoding, then interpret the output.

## Key Ideas

### 1. Logic can be a calculus of signs

Boole defines a sign as an arbitrary mark with a fixed interpretation that can be combined with other signs under fixed laws (Chapter II, p. 18). He proposes literal symbols such as $x$ and $y$, operation signs, and equality as the raw material for rational discourse.

A symbol need not resemble what it denotes. Correctness comes from a stable interpretation and valid rules of composition. This separates representation from operation—the same separation required by formal logic, programming languages, and computational graphs.

### 2. Symbols denote classes, and juxtaposition means simultaneous membership

If $x$ denotes white things and $y$ denotes sheep, then $xy$ denotes white sheep (Chapter II, p. 20). In modern set notation,

$$
xy \longleftrightarrow X\cap Y.
$$

The operation is commutative, $xy=yx$, because selecting things that satisfy both descriptions does not depend on their order. Complex meanings can thus be built by combining simpler predicates according to a formal rule.

### 3. The characteristic law is idempotence

Applying the same selection twice adds no new condition. Boole writes (Chapter II, p. 22)

$$
x^2=x.
$$

This is where logic parts from ordinary numerical algebra. Multiplying a number by itself usually changes it; intersecting a class with itself does not. The equation records the semantics of repeated classification.

Boole observes that, among ordinary numbers, only 0 and 1 satisfy $x^2=x$. He therefore compares the algebra of logic with an algebra whose variables admit only those two values (Chapter II, pp. 26–27):

```text
0 -> absence from the class / false
1 -> membership in the class / true
```

For Boole, the primary interpretation is usually class membership rather than today's truth-functional proposition variables. The binary bridge is real, but modern gate notation should not be projected backward unchanged.

### 4. Universe, emptiness, and complement enter the algebra

Boole includes “nothing” and the “universe” among classes (Chapter II, p. 20). With `1` as the universe of discourse, $1-x$ denotes everything not belonging to $x$. Products such as $x(1-y)$ denote things in $X$ but not in $Y$.

These operations provide ingredients later presented as AND and NOT. They also impose an essential modeling discipline: every complement is relative to a chosen universe.

### 5. Logical equations can be transformed and variables eliminated

Once premises are translated into equations, Boole applies algebraic expansion, reduction, and elimination to remove symbols representing intermediate classes. The objective is a general method: derive consequences about requested terms without rehearsing each argument as a named syllogism.

Inference becomes procedural. A conclusion is produced by transformations that preserve represented constraints.

### 6. Expansion decomposes every binary case

Boole evaluates an expression under $x=1$ and $x=0$, and for several symbols across combinations of presence and absence. In modernized form,

$$
f(x)=f(1)x+f(0)(1-x).
$$

For two variables the constituents are

$$
xy,\quad x(1-y),\quad (1-x)y,\quad (1-x)(1-y).
$$

Each term describes one mutually exclusive assignment. This is an ancestor of truth-table enumeration, disjunctive normal form, and the Shannon expansion used in switching theory and binary decision diagrams.

### 7. Boole's `+` is not simply modern inclusive OR

Historical accuracy matters. In *Laws of Thought*, $x+y$ is directly interpretable as aggregation when the classes are disjoint (Chapter II, pp. 22–24). Boole also permits formally uninterpretable intermediate expressions and recovers meaning at the end.

Modern Boolean algebra defines join $x\lor y$ for overlapping classes and obeys $x\lor x=x$. That cleaner algebra emerged through later work by Jevons, Peirce, Schröder, Huntington, and others. Boole founded the algebraic tradition in logic; he did not hand us today's gate algebra unchanged.

## Historical Context

Boole's shorter *Mathematical Analysis of Logic* appeared in 1847; *Laws of Thought* expanded the program in 1854 and connected it to probability. His departure from traditional logic was methodological: instead of cataloguing valid syllogisms, he sought general algorithms for arguments of arbitrary complexity.

The downstream path was not instantaneous or direct:

- later algebraic logicians reshaped Boole's class calculus into modern Boolean algebra;
- propositional logic gave truth-functional operations a cleaner formal syntax;
- Claude Shannon's 1937 MIT thesis (published in 1938; MIT repository record dated 1940) showed that Boolean equations could analyze and synthesize relay switching circuits;
- McCulloch and Pitts (1943) treated neural firing events as propositions and constructed threshold networks realizing temporal logical expressions.

Boole supplied the algebraic ancestor. Shannon supplied the switching interpretation. McCulloch and Pitts supplied the neural interpretation.

## Lineage

```text
Aristotelian syllogistic
(valid verbal forms, but no general symbolic algorithm)
        |
Viète and symbolic algebra
(letters + law-preserving transformations)
        |
Boole 1854
(classes become symbols; intersection becomes product;
 complement becomes 1-x; logical variables obey x²=x;
 binary cases can be expanded and eliminated)
        |
later Boolean algebra + propositional calculus
(total AND / OR / NOT operations over {0,1})
        |
Shannon 1938
(Boolean expressions become relay switching circuits)
        |
McCulloch-Pitts 1943
(logical expressions become threshold-neuron circuits)
        |
Hebb 1949 -> Rosenblatt 1958
(fixed logical networks acquire plastic and trainable weights)
        |
modern neural networks
(continuous optimization around compositional computation)
```

- **Built on:** [[../02-Algebraic-Revolution/Viète - Symbolic Notation 1591|Viète 1591]], because reasoning needed symbols and transformation rules before it could become a calculus.
- **Solved the next problem:** logic lacked a general algebraic procedure for representing and eliminating conditions in complex arguments.
- **Enabled:** Boolean algebra, truth tables, switching theory, digital circuits, formal verification, and the logical substrate used by [[McCulloch-Pitts - Neural Networks as Logical Calculus 1943|McCulloch–Pitts 1943]].
- **Did not yet solve:** physical implementation, computability, learning from examples, uncertainty under unknown distributions, or optimization of continuous parameters.

## Why It Matters for Mathematics

Boole demonstrated that algebra is not merely generalized arithmetic. Symbols may denote classes or logical conditions, and their laws can follow from that interpretation. This moves toward abstract algebra: operations are characterized by structure, not resemblance to counting.

His idempotent law exposes why different structures require different algebras. Over ordinary numbers, $x^2=x$ is exceptional. Over logical classes, it is universal. Choosing the correct algebra is part of modeling.

Expansion over all 0/1 assignments also turns reasoning into finite case decomposition. The same shape recurs in truth tables, SAT solving, decision trees, binary decision diagrams, circuit synthesis, and exact inference over discrete variables.

## Why It Matters for AI

### AI first needed computation that could be composed

McCulloch and Pitts could identify a firing event with a proposition because logic had become symbolic and compositional. Their networks implement conjunction, disjunction, inhibition, delay, and recurrence as arrangements of simple units. Boole is the upstream step that makes such operations algebraic objects rather than sentences.

### Boolean structure remains inside modern models

Contemporary neural networks usually train with real-valued activations and gradients, not strict 0/1 logic. Yet Boolean structure remains in token and attention masks, threshold decisions, sparse routing and gating, neuro-symbolic systems, formal verification, and the circuit implementation of every accelerator.

### Representation precedes optimization

Boole's durable lesson is architectural:

```text
choose symbols -> define semantics -> compose lawful operations
               -> derive output -> interpret it
```

Backpropagation later answers how to adjust continuous parameters inside a composed computation. It does not replace the prior requirement that the computation have a precise compositional form.

### The limitation points to learning

Boolean calculation derives consequences from premises encoded by a human. It does not discover useful predicates from data, tolerate graded ambiguity by itself, or revise its rules through error:

```text
Boole:             how can reasoning be calculated?
McCulloch-Pitts:   how can a network realize that calculation?
Hebb / Rosenblatt: how can experience alter the network?
backpropagation:   how can error assign credit through many layers?
```

That is why Boole belongs in the route to modern AI: before intelligence could be learned, part of reasoning had to become mechanically representable.

## References

- Boole, G. (1854). *An Investigation of the Laws of Thought*. Local primary source: `corpus/raw/library-papers/boole-laws-of-thought-1854.pdf`.
- Project Gutenberg ebook 15114: `https://www.gutenberg.org/ebooks/15114`.
- Burris, S. “George Boole.” *Stanford Encyclopedia of Philosophy*: `https://plato.stanford.edu/entries/boole/`.
- Shannon, C. E. *A Symbolic Analysis of Relay and Switching Circuits*. MIT record: `https://dspace.mit.edu/handle/1721.1/11173`.
- [[McCulloch-Pitts - Neural Networks as Logical Calculus 1943]]
- [[Hebb - Learning as Weight Change 1949]]

## Status

- [x] Primary source present at `corpus/raw/library-papers/boole-laws-of-thought-1854.pdf`
- [x] Downloaded with `curl -L -o`
- [x] Verified with `file` as a PDF document, version 1.5
- [x] Complete source checked: 343 pages
- [x] Boole's original class algebra distinguished from modern Boolean algebra
- [x] Wired into the chain before McCulloch–Pitts
- [ ] Natural next link: Shannon 1938 — Boolean algebra becomes an engineering calculus for switching circuits

---

*Before logic could be wired into a neuron, it had to become something algebra could touch.*
