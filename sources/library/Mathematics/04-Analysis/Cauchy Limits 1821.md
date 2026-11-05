# Augustin-Louis Cauchy — Limits and Convergence (1821)

#person #paper #milestone #mathematics #analysis #limits #convergence #continuity

**Person:** Augustin-Louis Cauchy (1789–1857)  
**Work:** *Cours d’analyse de l’École royale polytechnique, I.re partie: Analyse algébrique* (1821)  
**Key contribution:** Cauchy reorganized analysis around limits: infinitesimals became variable quantities tending to zero, convergence became a property of partial sums, continuity became preservation of infinitesimal change, and the tail condition now bearing his name made convergence testable without first knowing the limit.  
**Paper:** `corpus/raw/library-papers/cauchy-cours-analyse-1821.pdf`  
**Source status:** Verified scan of the original French edition, downloaded from the Internet Archive. `file` identifies it as a PDF document; `pdfinfo` reports 604 scanned pages.  
**Public source:** <https://archive.org/details/TO0E037871_TO0324_PNI-1898_000000>

---

## Title

**Infinite calculation needed a rule for when “approaches” becomes a mathematical fact.**

Euler had made functions and infinite series into working instruments. Fourier then claimed that broad temperature profiles could be reconstructed from infinitely many trigonometric modes. Their successes created a problem more serious than a missing technique:

> What does it mean for a changing quantity or an infinite sum to reach a value, and what must be proved before finite algebra may be extended to an infinite process?

Cauchy’s answer was not yet the final modern foundation. He still reasoned with variable quantities and infinitesimal increments, and he did not construct the completeness of the real numbers on which his convergence criterion depends. But the *Cours d’analyse* made limit, convergence, continuity, and convergence tests the explicit grammar of analysis.

This is the missing hinge from **Euler’s formal series and Fourier’s representations** to **Dirichlet’s conditional Fourier theorem, Riemann’s integral, Weierstrass’s quantified rigor, and Dedekind’s construction of the real continuum**.

---

## Paper / Source Reference

**Cauchy, A.-L. (1821).** *Cours d’analyse de l’École royale polytechnique; I.re Partie. Analyse algébrique*. Paris: Imprimerie Royale / Debure frères.

Local source:
- `corpus/raw/library-papers/cauchy-cours-analyse-1821.pdf`

Public sources:
- Internet Archive scan and metadata: <https://archive.org/details/TO0E037871_TO0324_PNI-1898_000000>
- MacTutor historical biography: <https://mathshistory.st-andrews.ac.uk/Biographies/Cauchy/>
- Annotated English translation record: <https://doi.org/10.1007/978-1-4419-0549-9>

The local primary text was checked directly at the preliminary definition of limit, Chapter II on infinitesimals and continuity, and Chapter VI on convergent series and their tails.

---

## What Problem Made This Necessary?

### 1. Finite algebra does not automatically survive infinity

For a finite sum, regrouping terms, multiplying expressions, or passing an operation through the sum is usually routine. An infinite series is different. It is shorthand for the behavior of a sequence of finite partial sums:

\[
s_n=u_0+u_1+\cdots+u_{n-1}.
\]

Euler’s symbolic fluency often produced correct identities, but the period lacked one systematic standard for deciding when such an infinite expression possessed a sum. Cauchy’s preface says that he sought the rigor demanded in geometry rather than relying on the “generality of algebra.” The target was precisely the habit of extending finite rules merely because the symbols looked the same.

### 2. “Infinitely small” needed operational meaning

Infinitesimals had long powered calculus, but they could sound like mysterious fixed numbers smaller than every ordinary magnitude. Cauchy instead described an infinitesimal dynamically: it is a **variable quantity whose successive numerical values tend to zero**.

That move ties the infinitesimal to a limiting process. The foundation is no longer a special static entity; it is the behavior of a variable.

### 3. Fourier made convergence urgent rather than optional

Fourier coefficients can be calculated from a function, but coefficient formulas alone do not show that the resulting trigonometric partial sums converge to that function. Heat theory therefore exposed the exact weakness in formal series reasoning: a useful representation still needs a truth condition.

Cauchy made convergence central, but his own attempted Fourier argument did not settle the problem. Dirichlet’s 1829 paper would open by explaining why that proof was insufficient and would replace it with a theorem under explicit regularity assumptions.

---

## Key Ideas

### 1. A limit is defined through arbitrary closeness

In the preliminary section, Cauchy says that when the successive values of a variable approach a fixed value indefinitely, eventually differing from it “as little as one wishes,” that fixed value is the limit.

In modern sequence notation, the intended structure is:

\[
x_n\to L
\quad\text{when eventually}\quad
|x_n-L|\text{ is smaller than any prescribed positive tolerance.}
\]

Cauchy’s wording is not yet the canonical modern statement

\[
\forall \varepsilon>0\;\exists N\;\forall n\ge N:
|x_n-L|<\varepsilon,
\]

but the essential quantifier pattern is visible: first prescribe the closeness, then go sufficiently far in the process.

Cauchy also introduced `lim.` as working notation. A limit became an object one could name, calculate, and prove properties about—not merely an intuition attached to motion.

### 2. Infinitesimals become null sequences

Cauchy defines an infinitely small quantity as a variable whose numerical value decreases indefinitely so as to converge to zero. In modern language, if

\[
\alpha_n\to 0,
\]

then \(\alpha_n\) is a null sequence.

He also compares orders of infinitesimals by ratios. If \(\alpha\to0\) and

\[
\frac{\beta(\alpha)}{\alpha^p}\to c,
\qquad 0<|c|<\infty,
\]

then \(\beta\) behaves like an infinitesimal of order \(p\). This is an ancestor of asymptotic notation: local behavior is classified by relative rate of vanishing.

### 3. Continuity means infinitesimal input change produces infinitesimal output change

For a function \(f\), Cauchy considers an infinitesimal increment \(\alpha\) of the input and the corresponding increment

\[
f(x+\alpha)-f(x).
\]

He calls \(f\) continuous when this output difference becomes infinitesimal with \(\alpha\), for each point in the interval under discussion. In modern pointwise form:

\[
\lim_{h\to0}\bigl(f(x+h)-f(x)\bigr)=0,
\]

or equivalently,

\[
\lim_{y\to x}f(y)=f(x).
\]

This makes continuity a claim about limiting behavior, not merely about whether a curve can be drawn without lifting the pencil.

### 4. A series is the limit of its partial sums

Cauchy defines an infinite series through the sequence of partial sums. If \(s_n\) approaches a fixed limit \(s\), the series converges and \(s\) is its sum; otherwise it diverges and has no sum.

That definition blocks a common symbolic mistake. The written expression

\[
u_0+u_1+u_2+\cdots
\]

is not automatically a number. It denotes a number only when its finite approximations converge.

### 5. Convergence can be tested by the tails

Cauchy states that convergence requires—and, in his number system, is ensured by—the tails becoming smaller than any assignable bound. For a series \(\sum u_k\), this means that for sufficiently large \(n\), every finite tail

\[
u_n+u_{n+1}+\cdots+u_{n+p}
\]

must be arbitrarily small, however many later terms \(p\) are included.

In modern notation:

\[
\forall \varepsilon>0\;\exists N\;\forall m>n\ge N:
\left|\sum_{k=n+1}^{m}u_k\right|<\varepsilon.
\]

Equivalently, the partial sums must satisfy

\[
|s_m-s_n|<\varepsilon
\]

for all sufficiently large \(m,n\). This is the **Cauchy criterion**.

Its conceptual power is enormous: one can test whether approximations settle together without knowing the answer in advance. But there is a hidden foundation. “Every Cauchy sequence converges” is true in a complete space such as \(\mathbb R\), not in \(\mathbb Q\). Cauchy used the criterion before Dedekind and Cantor supplied an explicit arithmetic account of the complete real continuum.

### 6. The harmonic series separates vanishing terms from convergence

Cauchy emphasizes that \(u_n\to0\) is necessary but not sufficient for \(\sum u_n\) to converge. For the harmonic series,

\[
1+\frac12+\frac13+\frac14+\cdots,
\]

the individual terms vanish, yet blocks of terms retain substantial mass. For example,

\[
\frac{1}{n+1}+\cdots+\frac{1}{2n}
\]

stays bounded below by roughly \(1/2\). The tails therefore do not become arbitrarily small, so the series diverges.

This is exactly the kind of distinction formal manipulation obscures: local smallness of each term does not imply global convergence of their accumulation.

### 7. Cauchy’s own overreach exposed the next missing concept

In the *Cours d’analyse*, Cauchy stated that a convergent series of continuous functions has a continuous sum. Under merely pointwise convergence, this is false. Abel soon supplied counterexamples; later analysis isolated **uniform convergence** as a sufficient condition for continuity to pass to the limit.

This failure is not incidental to the lineage. It reveals that “converges” is not one property:

- pointwise convergence controls each input separately;
- uniform convergence controls one error bound across the whole domain;
- later analysis adds convergence in measure, almost everywhere, and in \(L^p\).

Cauchy made convergence the question. His incomplete sum theorem showed that the question needed finer answers.

---

## Historical Context

The *Cours d’analyse* grew from Cauchy’s teaching at the École Polytechnique and was published in Paris in 1821. It was designed to rebuild algebraic analysis with proof standards closer to Euclidean geometry.

The historical sequence is not perfectly chronological in publication. Fourier had presented his heat work to the Academy in 1807, Cauchy published the *Cours* in 1821, and Fourier’s major treatise appeared in 1822. The conceptual sequence is nevertheless clear:

- eighteenth-century analysts made powerful use of infinite expansions;
- Fourier made broad spectral representation indispensable to a physical theory;
- Cauchy made limits and convergence explicit objects of proof;
- Dirichlet then supplied conditions under which Fourier reconstruction actually converges.

Cauchy did not simply replace infinitesimals with a modern epsilon–delta system. His language retained variable quantities and infinitesimal increments, while many proofs used tolerance-style inequalities. Weierstrass and his school later standardized the static quantified formulations, and Dedekind/Cantor clarified the real-number foundation that Cauchy’s criteria tacitly needed.

The 1821 volume is algebraic analysis. Cauchy’s later 1823 *Résumé des leçons données à l’École royale polytechnique sur le calcul infinitésimal* developed differential and integral calculus. Keeping those works distinct matters: this note is about the limit-and-convergence grammar established in the *Cours*, not about attributing every later piece of rigorous calculus to one book.

---

## Lineage

```text
Newton & Leibniz 1666–1686
  → local rate and global accumulation become a general calculus
Euler 1748
  → functions and infinite series become the language of analysis
Fourier 1807–1822
  → arbitrary profiles are represented by infinitely many modes
  → but formal coefficients do not prove reconstruction
Cauchy 1821
  → limit, infinitesimal, continuity, and series convergence become explicit
  → the tail criterion tests convergence without knowing the limit
  → the false sum theorem exposes pointwise vs uniform convergence
Dirichlet 1829
  → Fourier convergence receives explicit hypotheses and a pointwise limit
Riemann / Weierstrass
  → integration and quantifiers are sharpened; modes of convergence separate
Dedekind / Cantor
  → the complete real continuum is constructed and its infinity analyzed
Lebesgue
  → integration and convergence become stable under measure-theoretic limits
Modern AI
  → optimization, approximation, and learning claims must specify what converges, to what, and in which sense
```

### What came before

- [[Euler Infinite Series 1748|Euler]] made infinite expansion a central representation and computational method.
- [[Fourier - Heat Equation and Trigonometric Representation 1807-1822|Fourier]] made the equality between a function and an infinite spectral representation a concrete, urgent problem.
- [[../05-Newton-Leibniz/Newton-Leibniz - Fundamental Theorem of Calculus|Newton–Leibniz]] supplied the calculus whose foundations Cauchy sought to discipline.

### What Cauchy enabled—and what remained broken

- [[Dirichlet - Fourier Series Convergence 1829|Dirichlet]] could ask for exact hypotheses and identify the value reconstructed at a jump.
- [[Riemann - Integration and Trigonometric Series 1854|Riemann]] could define integrability through convergence of tagged sums and confront discontinuous functions.
- [[Weierstrass Epsilon-Delta 1861|Weierstrass]] could standardize quantified limit arguments and distinguish pointwise from uniform control.
- [[../08-Set-Theory-Infinity/Dedekind - Real Numbers and Cuts 1872|Dedekind]] could repair the hidden completeness assumption beneath the Cauchy criterion.
- [[../11-Optimization-Theory/Cauchy - Steepest Descent 1847|Cauchy’s 1847 steepest descent]] later turned derivatives into an iterative convergence process, creating a separate direct bridge to machine-learning optimization.

The gap closed by this note is:

**Euler and Fourier can manipulate infinite representations → Cauchy defines what it means for finite approximations to settle → Dirichlet and later analysts determine the assumptions and mode under which the limit is truthful.**

---

## Why It Matters for AI

### 1. Training is a claim about a sequence, not a single update

An optimizer produces iterates

\[
\theta_0,\theta_1,\theta_2,\ldots.
\]

Saying that loss decreased once is not a convergence result. A mathematical claim must identify the candidate limit, the metric or objective being controlled, the assumptions, and whether the sequence itself, its gradients, its averages, or only a subsequence converges.

Cauchy’s shift from symbolic expression to partial-sum sequence is the same shift required to reason clearly about optimization trajectories.

### 2. The Cauchy criterion underlies numerical stopping logic

In computation, the true solution is often unknown. What can be observed is whether successive approximations are settling:

\[
\|x_m-x_n\|<\varepsilon.
\]

Iterative solvers, fixed-point methods, numerical linear algebra, and optimization all inherit this logic. Practical stopping tests such as small parameter updates are imperfect proxies for a Cauchy condition.

The completeness caveat matters too. A sequence can look internally consistent while leaving the representational space in which the algorithm is searching. Choosing a closed, complete function or parameter space is part of making a convergence proof valid.

### 3. Vanishing updates do not imply a finite accumulated effect

The harmonic series is a permanent warning for learning-rate schedules. Individual steps \(\alpha_t\to0\) can still accumulate without bound. Conversely, stochastic approximation often deliberately requires

\[
\sum_t \alpha_t=\infty,
\qquad
\sum_t \alpha_t^2<\infty,
\]

so that updates remain capable of reaching the target while their noise has finite accumulated variance.

Robbins–Monro is therefore not detached from classical analysis. It is a statistical descendant of Cauchy’s distinction between small terms and convergent accumulation.

### 4. Pointwise success is not uniform reliability

A model sequence may converge for every fixed input while lacking one shared error guarantee over the input domain. This mirrors Cauchy’s false continuity-of-the-sum theorem.

The distinction appears throughout AI:

- pointwise prediction convergence versus worst-case robustness;
- empirical loss convergence versus uniform generalization;
- per-prompt success versus reliability across a deployment distribution;
- convergence of finite-width or finite-depth models versus stable behavior over all inputs.

A mode of convergence is part of the result, not fine print.

### 5. Function approximation needs truth conditions

Fourier series, kernel expansions, neural operators, and neural networks all represent complex functions through finite parameterizations or truncations. Cauchy’s central demand survives:

> Do the approximations converge, and does their limit equal the intended function?

Modern approximation and learning theory add norms, probability, capacity, and finite-sample bounds, but they are answering Cauchy’s question in richer spaces.

### 6. Convergence of values, parameters, and behavior are different

In overparameterized models, many parameter vectors can implement nearly the same function. A loss may converge while parameters drift; parameters may stabilize at a saddle; predictions may converge in probability but fail on rare cases.

Cauchy’s legacy is not a guarantee that these notions coincide. It is the discipline of refusing to say “converges” until the object and the sense of convergence are named.

---

## The Core Intuition

Cauchy turned infinity from a license to continue algebra forever into a demand about finite approximations:

> An infinite object earns its value only when its finite stages settle under a stated criterion.

That sentence is foundational for modern AI. Every training run, truncated representation, numerical solver, and asymptotic guarantee lives or dies by what settles, in which space, under which assumptions.

---

## References

### Primary source
- Cauchy, A.-L. (1821). *Cours d’analyse de l’École royale polytechnique; I.re Partie. Analyse algébrique*. Paris: Imprimerie Royale / Debure frères. Local PDF: `corpus/raw/library-papers/cauchy-cours-analyse-1821.pdf`. Internet Archive: <https://archive.org/details/TO0E037871_TO0324_PNI-1898_000000>.

### Translation and historical context
- Bradley, R. E., & Sandifer, C. E. (2009). *Cauchy’s Cours d’analyse: An Annotated Translation*. Springer. DOI: <https://doi.org/10.1007/978-1-4419-0549-9>.
- O’Connor, J. J., & Robertson, E. F. “Augustin-Louis Cauchy.” *MacTutor History of Mathematics Archive*. <https://mathshistory.st-andrews.ac.uk/Biographies/Cauchy/>.
- Dirichlet, P. G. L. (1829). “Sur la convergence des séries trigonométriques…” Local PDF: `corpus/raw/library-papers/0806.1294.pdf`.

---

## Status

- [x] Original 1821 source downloaded: `corpus/raw/library-papers/cauchy-cours-analyse-1821.pdf`
- [x] Verified with `file`: PDF document, version 1.5
- [x] Verified with `pdfinfo`: 604 pages, unencrypted
- [x] Primary-source passages checked: limit, infinitesimal, continuity, series convergence, tail criterion, harmonic-series counterexample
- [x] Distinguishes the 1821 analysis work from Cauchy’s 1847 steepest-descent paper
- [x] No orphan: linked to Euler, Fourier, Dirichlet, Riemann, Weierstrass, Dedekind, optimization, the MOC, and LINEAGE.md

---

*Euler taught analysis to calculate with infinity. Cauchy made infinity answer for the behavior of its finite stages.*
