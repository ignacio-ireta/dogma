# Leonhard Euler — Functions and Infinite Series (1748)

#person #paper #milestone #mathematics #analysis

**Person:** Leonhard Euler (1707–1783)  
**Work:** *Introductio in analysin infinitorum*, Volume I (1748), Eneström E101  
**Paper:** `corpus/raw/library-papers/euler-introductio-analysis-infinite-1748.pdf`  
**Source status:** Verified PDF of the original Latin volume, 353 pages, downloaded from the Euler Archive.  
**Euler Archive:** <https://scholarlycommons.pacific.edu/euler-works/101/>

---

## Title

**The function becomes the object, and an infinite series becomes a way to represent it.**

Newton and Leibniz had built a calculus for tangent, velocity, area, and accumulation. But their methods were still tied to particular curves and geometric or mechanical problems. Mathematics could differentiate a given expression without yet having a unified language for the *things being differentiated*.

That was the problem Euler inherited:

> How can calculus become a general science rather than a collection of techniques for individual curves?

His answer was to place the **function** at the center of analysis and to treat infinite expansions as operational representations of functions. This move made Fourier's later spectral representation conceivable—and made Cauchy's later demand for convergence unavoidable.

---

## The Problem That Made Euler Necessary

The seventeenth-century calculus had solved the local/global problem:

- differentiation extracted an instantaneous rate;
- integration accumulated local contributions;
- the Fundamental Theorem related the two.

But the subject still lacked a stable organizing object. A curve might be given geometrically, a logarithm through a quadrature, a trigonometric quantity through a circle, and a differential equation through a physical system. These methods worked, but they did not yet look like instances of one theory.

Euler reorganized the landscape around dependence between variable quantities. In the opening chapter of the *Introductio*, a function is an analytic expression composed from a variable and constants. That definition is narrower than the modern idea of an arbitrary mapping, but the conceptual promotion is decisive: the curve is no longer primary. The curve is one representation of a function.

The second problem was computation. Transcendental functions such as exponentials, logarithms, and trigonometric functions could not be handled by finite algebra alone. Infinite series, products, and continued fractions offered a common language in which such functions could be calculated and compared.

So Euler's two moves belong together:

1. make the **function** the central object;
2. make an **infinite expansion** a calculable representation of that object.

---

## Key Ideas

### 1. Analysis becomes the study of functions

Euler defines a function of a variable quantity through an analytic expression involving that variable and constants. In modern notation, the basic relation is written

\[
y=f(x).
\]

This lets one speak about algebraic, exponential, logarithmic, and trigonometric dependence in the same grammatical form. The importance is not that Euler already possessed the modern set-theoretic definition—he did not—but that he made functions the *subject matter* of analysis.

Before this shift, sine and cosine could be regarded mainly as lengths associated with a circle. In Euler's treatment, they become functions that can be transformed, expanded, compared, and combined with other functions.

### 2. A function can be represented by an infinite process

Euler systematically uses expansions of the form

\[
f(x)=a_0+a_1x+a_2x^2+a_3x^3+\cdots.
\]

The series is not merely a decorative identity. It turns a difficult function into a sequence of finite polynomial approximations. Truncate after finitely many terms and one obtains something computable; add terms and one hopes to improve the approximation.

For the exponential function, the emerging grammar is

\[
e^x=1+x+\frac{x^2}{2!}+\frac{x^3}{3!}+\cdots,
\]

while logarithmic and trigonometric functions receive related expansions. Infinite algebra becomes an engine for evaluating and relating functions that finite algebra cannot express.

This is the ancestor of a huge modern habit: choose a family of basis functions, represent a target by coefficients, and compute through a finite truncation.

### 3. Exponential and trigonometric functions enter one system

The *Introductio* gives the complex-exponential relation

\[
e^{ix}=\cos x+i\sin x,
\]

from which trigonometric identities can be derived algebraically. Exponentials and rotations are no longer separate mathematical species. They are two views of the same analytic structure.

That unification later becomes central to Fourier analysis: oscillations can be written as complex exponentials, manipulated as algebra, and recombined into signals.

### 4. Representation creates a new question: when is it true?

Euler's strength was fearless formal manipulation. But infinite sums are not finite sums with an ellipsis added. Several operations that are harmless for finite expressions—rearranging terms, multiplying series, exchanging limits, or assigning values outside a convergence domain—require hypotheses in the infinite case.

Euler often obtained correct and astonishing results, but his framework did not yet supply a general theory answering questions such as:

- Does the sequence of partial sums converge?
- On which values of \(x\) does it converge?
- Does it converge to the function it is supposed to represent?
- Can a limit be exchanged with differentiation or integration?
- If functions converge pointwise, is continuity preserved?

Euler did not fail by leaving these questions open. He made them impossible to ignore. Once functions were represented by infinite processes, **convergence became a foundational problem rather than a technical afterthought**.

### 5. A function is not yet fully separated from its formula

Euler's function concept remained expression-centered. A function was largely identified with an analytic formula, and his period's terms “continuous” and “discontinuous” did not have their modern meanings. A piecewise or freely drawn curve strained this definition.

This limitation matters to the lineage. Fourier would claim that very broad initial temperature profiles—even irregular or piecewise ones—could be represented by trigonometric series. That claim forced mathematics to distinguish:

- a function from one formula used to describe it;
- a representation from the object represented;
- formal expansion from proven convergence.

Euler provides the language. Fourier pushes it beyond its safe domain. Cauchy, Dirichlet, Riemann, Weierstrass, Cantor, and Lebesgue then rebuild the foundations.

---

## Historical Context

Euler wrote the *Introductio* in the 1740s and published it in two volumes in 1748. Volume I develops variables, functions, transformations, infinite series, exponentials, logarithms, trigonometric functions, products, and continued fractions. Volume II turns toward analytic geometry.

The book is striking because it prepares analysis without making differentiation or integration the main organizing principle. It takes the calculus inherited from Newton, Leibniz, and the Bernoullis and builds the functional language in which later analysis will operate.

A historical caveat is essential: Euler did not single-handedly invent every ingredient. Leibniz and Johann Bernoulli had already used the word *function*, and power-series methods preceded 1748. Euler's achievement was synthesis, centrality, and scale. He made the function-and-series viewpoint into a coherent mathematical world.

---

## Lineage

```text
Newton & Leibniz 1666–1686
(local rate ↔ global accumulation; a general calculus)
    ↓ calculus needs a common object and a computational language
Euler 1748
(functions become central; infinite series represent transcendental dependence)
    ↓ can broad or irregular functions really equal infinite expansions?
Fourier 1807–1822
(arbitrary temperature profiles represented by trigonometric modes)
    ↓ formal success is not a convergence theorem
Cauchy 1821 / Dirichlet 1829
(explicit convergence; hypotheses for Fourier reconstruction)
    ↓ pointwise convergence and interval integration remain insufficient
Riemann 1854 / Weierstrass 1860s
(integration tests, ε–δ rigor, modes of convergence)
    ↓ exceptional sets and limiting operations need a stronger framework
Cantor / Lebesgue
(sets of points, measure, integration stable under limits)
```

### What came before

- [[../05-Newton-Leibniz/Newton-Leibniz - Fundamental Theorem of Calculus|Newton–Leibniz]] made differentiation and integration general inverse operations.
- Algebraic notation made general expressions manipulable.
- The Bernoullis and earlier analysts had already introduced function language and series techniques.

### What Euler enabled

- [[Fourier - Heat Equation and Trigonometric Representation 1807-1822|Fourier]] could treat a function as something decomposable into modes.
- [[Cauchy Limits 1821|Cauchy]] had a concrete crisis to repair: infinite calculations needed definitions of limit and convergence.
- [[Dirichlet - Fourier Series Convergence 1829|Dirichlet]] could ask exactly which functions are reconstructed by a trigonometric series and what happens at jumps.
- [[Weierstrass Epsilon-Delta 1861|Weierstrass]] could separate intuitive closeness from quantified proof.
- Modern approximation theory could treat representation as the search for coefficients in a function family.

---

## Why It Matters for AI

Modern AI inherits Euler's move before it inherits any particular algorithm: **a model is a parameterized function**.

A neural network is written as

\[
f_\theta(x),
\]

where \(x\) is an input and \(\theta\) collects adjustable parameters. Training changes \(\theta\) so that this function approximates a desired relation. That notation carries Euler's functional viewpoint into machine learning almost unchanged.

The series viewpoint also survives:

- **Approximation:** finite models approximate more complicated functions, just as truncated series approximate their limits.
- **Features and bases:** polynomial features, Fourier features, kernels, and positional encodings represent data through chosen function families.
- **Spectral methods:** Fourier modes turn coupled structure into independently manipulable components.
- **Function classes:** learning theory asks not only whether one function fits the data, but what an entire hypothesis class can represent and generalize.
- **Convergence:** optimization and statistical learning require proof that finite procedures approach a meaningful limit rather than merely producing plausible numbers.

Neural networks are not power series, and it would be sloppy to call Euler their inventor. The deeper inheritance is structural: difficult dependence becomes tractable when represented inside a parameterized family, then approximated by finite computation.

Euler gave analysis the object. Fourier gave it modes. Cauchy and Weierstrass gave it truth conditions. Modern AI turns the object into something learnable.

---

## Sources

### Primary source
- Euler, Leonhard. *Introductio in analysin infinitorum*, Volume I. Lausanne: Marcum-Michaelem Bousquet, 1748. Euler Archive E101. Local PDF: `corpus/raw/library-papers/euler-introductio-analysis-infinite-1748.pdf`

### Historical context and verification
- Euler Archive, E101 record and content summary: <https://scholarlycommons.pacific.edu/euler-works/101/>
- MacTutor History of Mathematics, “The function concept”: <https://mathshistory.st-andrews.ac.uk/HistTopics/Functions/>
- MacTutor History of Mathematics, “Leonhard Euler”: <https://mathshistory.st-andrews.ac.uk/Biographies/Euler/>

---

## Status

- [x] Original source downloaded
- [x] PDF verified as a 353-page PDF document
- [x] Source path recorded in the note
- [x] Wired into Newton–Leibniz → Euler → Fourier → rigorous analysis

---

*Euler's great move was not merely summing more series. It was deciding that functions were the country, and infinite expansions were one way to draw its maps.*
