# Bernhard Riemann — Which Discontinuous Functions Can Be Integrated? (1854)

#person #paper #milestone #analysis #integration #fourier-series #convergence

**Person:** Georg Friedrich Bernhard Riemann (1826–1866)  
**Work:** *Ueber die Darstellbarkeit einer Function durch eine trigonometrische Reihe* (submitted as a Habilitation dissertation in 1854; published posthumously in 1867)  
**Key contribution:** Riemann defined the definite integral as the common limit of all tagged partition sums as the largest interval shrinks to zero, then investigated necessary and sufficient conditions for that limit to exist. Integration could now include many discontinuous functions—but its limits also became visible.  
**Paper:** `corpus/raw/library-papers/riemann-trigonometric-series-integration-1854.pdf`  
**Source note:** The local file is a 46-page public-domain scan of the standalone 1867 Göttingen publication of Riemann’s 1854 dissertation, downloaded from the Internet Archive. It was verified with `file` as a PDF document (version 1.4). The dissertation’s integration discussion begins under “Ueber den Begriff eines bestimmten Integrals und den Umfang seiner Gültigkeit.”

---

## Title

Dirichlet had proved when a broad class of Fourier series converges. Riemann had to ask whether the integral used to define those series was broad enough for the functions the theory itself was creating.

Fourier analysis had broken the old identification of a function with one smooth analytic formula. Initial temperature profiles could be piecewise, measured, or discontinuous. [[Dirichlet - Fourier Series Convergence 1829|Dirichlet]] then made spectral reconstruction conditional: bounded piecewise-monotone functions have Fourier partial sums that converge to the midpoint of their one-sided limits.

But this left a deeper problem hidden inside the coefficient formulas

$$
a_n=\frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\cos(nx)\,dx,
\qquad
b_n=\frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\sin(nx)\,dx.
$$

What exactly does the integral mean when $f$ is not continuous? Which discontinuities can the operation tolerate? When does every sufficiently fine finite approximation settle on one value, regardless of how the sample points were chosen?

That is the problem that made Riemann’s definition necessary:

> **Before asking whether a Fourier series reconstructs a function, analysis must know whether the function has Fourier coefficients at all.**

Riemann’s answer was neither “only continuous functions” nor “every bounded function.” He defined integration by a universal stability requirement over finer and finer tagged partitions. This enlarged the integral to many discontinuous functions while giving a precise test for where the enlargement stopped.

## Paper / Source Reference

**Riemann, B. (1854/1867).** *Ueber die Darstellbarkeit einer Function durch eine trigonometrische Reihe*. Habilitation dissertation submitted at Göttingen in 1854; published posthumously in *Abhandlungen der Königlichen Gesellschaft der Wissenschaften zu Göttingen*, vol. 13.

Local primary source:
- `corpus/raw/library-papers/riemann-trigonometric-series-integration-1854.pdf`

Public sources:
- Internet Archive scan: <https://archive.org/details/ueberdiedarstel00librgoog>
- Direct public-domain PDF: <https://archive.org/download/ueberdiedarstel00librgoog/ueberdiedarstel00librgoog.pdf>
- Encyclopedia of Mathematics, “Riemann integral”: <https://encyclopediaofmath.org/wiki/Riemann_integral>
- MacTutor, “Bernhard Riemann”: <https://mathshistory.st-andrews.ac.uk/Biographies/Riemann/>

The date needs care. The mathematics was written for Riemann’s 1854 Habilitation. The surviving work was first printed after his death, in 1867. This note uses **1854** for its place in the lineage and records **1867** for the source edition.

## What Problem Made This Necessary?

### 1. Newton and Leibniz gave integration a calculus role, not a general existence test

The [[../05-Newton-Leibniz/Newton-Leibniz - Fundamental Theorem of Calculus|fundamental theorem]] connected accumulated area to antiderivatives. That was revolutionary, but it encouraged a comfortable world in which functions were smooth enough to differentiate and integrate through formulas.

Fourier’s arbitrary profiles did not live comfortably in that world. A function might be given graphically, piecewise, or with jumps, yet still be physically meaningful. Analysis needed an integral defined without first assuming an antiderivative or one analytic expression.

### 2. Cauchy’s integral did not settle the full discontinuous case

[[Cauchy Limits 1821|Cauchy]] had defined integration through limits of sums and made convergence explicit, chiefly for continuous functions, with extensions for certain improper cases. But Fourier analysis made discontinuity central rather than exceptional.

Riemann therefore opens the relevant section with the foundational question:

> What is to be understood by $\int_a^b f(x)\,dx$?

He then asks the matching existence question: in which cases does a function admit integration, and in which cases does it not?

### 3. Dirichlet’s theorem had assumptions, but the integral beneath it needed its own domain

Dirichlet’s 1829 proof worked for bounded functions with finitely many discontinuities and finitely many maxima and minima on a period. Those conditions made his oscillatory-integral argument controllable.

Riemann wanted to go further. His dissertation asks not only when regularity implies a Fourier representation, but what can be inferred in the reverse direction from a trigonometric representation. To do that, he needed an integral capable of handling functions with infinitely many discontinuities—while still rejecting functions whose local oscillation never becomes negligible.

The immediate gap was therefore:

```text
Dirichlet: Fourier convergence for piecewise-regular functions
  → but “integrable” still lacks a sharp general test
Riemann: define integration through all fine tagged sums
  → admit many discontinuous functions and expose the boundary of the method
Lebesgue: replace interval control with measure of sets
  → integration becomes stable under much broader limits
```

## Key Ideas

### 1. A definite integral is the common limit of all fine tagged sums

Partition $[a,b]$ into intervals

$$
a=x_0<x_1<\cdots<x_n=b,
$$

write $\Delta x_i=x_i-x_{i-1}$, and choose an arbitrary sample point $\xi_i\in[x_{i-1},x_i]$. Form the finite sum

$$
S(P,\xi)=\sum_{i=1}^{n}f(\xi_i)\Delta x_i.
$$

Let the **mesh** of the partition be

$$
\lVert P\rVert=\max_i\Delta x_i.
$$

Riemann’s requirement is that there exist one number $A$ such that, however the intervals and sample points are chosen,

$$
S(P,\xi)\longrightarrow A
\qquad\text{as}\qquad
\lVert P\rVert\to 0.
$$

Then

$$
\int_a^b f(x)\,dx=A.
$$

In modern quantifiers: for every $\varepsilon>0$, there is a $\delta>0$ such that every tagged partition with $\lVert P\rVert<\delta$ satisfies

$$
\left|S(P,\xi)-A\right|<\varepsilon.
$$

The word **every** does the real work. It is not enough to find one convenient sequence of rectangles that approaches a value. The result must be stable against arbitrary fine choices of interval widths and sampling locations.

### 2. Integrability is a robustness statement

A partition sum contains two approximations:

1. replace the function on each interval by one sampled value;
2. replace continuous accumulation by a finite weighted sum.

If changing the sample point inside a tiny interval can still change the total appreciably, no unique integral exists. If all such choices become irrelevant as the mesh shrinks, the integral is well defined.

This makes the integral a statement of **approximation invariance**:

> Once resolution is fine enough, microscopic sampling choices must not alter the macroscopic total.

That is stronger and cleaner than identifying the integral with a picture of area.

### 3. Local oscillation determines whether the sums can agree

For each subinterval $I_i$, let

$$
\omega_i=\sup_{x\in I_i}f(x)-\inf_{x\in I_i}f(x)
$$

be the function’s oscillation there. The largest possible difference caused by choosing different tags is controlled by

$$
\sum_{i=1}^{n}\omega_i\Delta x_i.
$$

Riemann’s analysis shows that a bounded function is integrable precisely when this weighted total oscillation can be made arbitrarily small by sufficiently fine partitions.

He states the condition in terms of intervals where oscillation exceeds a fixed threshold $\sigma>0$: as the partition is refined, the total length of those badly oscillating intervals must become arbitrarily small.

This is the conceptual heart of the theory. Discontinuity by itself is not fatal. What matters is whether persistent oscillation occupies enough interval length to keep different sampling choices apart.

Later, after measure theory existed, Lebesgue gave the familiar equivalent criterion:

> A bounded function on $[a,b]$ is Riemann integrable if and only if its set of discontinuities has Lebesgue measure zero.

That modern sentence should not be projected backward as Riemann’s vocabulary. Riemann had the interval-oscillation condition; Lebesgue later supplied the set-measure formulation.

### 4. Infinitely many discontinuities need not destroy integrability

Riemann explicitly studies functions discontinuous infinitely often in every interval. He constructs an example whose jumps become increasingly small and shows that it remains integrable.

This breaks a tempting but false rule:

```text
continuous → integrable
therefore discontinuous → not integrable
```

The correct hierarchy is subtler:

- every continuous function on a closed bounded interval is Riemann integrable;
- many functions with finitely or infinitely many discontinuities are also integrable;
- some bounded functions oscillate too violently to be Riemann integrable.

The theory enlarges the function class without pretending that every bounded assignment of values can be accumulated coherently.

### 5. The Dirichlet function reveals the boundary

Consider

$$
f(x)=
\begin{cases}
1,&x\in\mathbb Q,\\
0,&x\notin\mathbb Q.
\end{cases}
$$

Every interval contains both rational and irrational numbers. Therefore every interval has oscillation $1$, no matter how small it becomes. Tagged sums can be forced to equal $b-a$ by choosing rational tags or $0$ by choosing irrational tags.

There is no common limit, so the function is not Riemann integrable.

This failure is instructive. The function is bounded and defined everywhere; what fails is not existence of values but stability under local sampling. [[../08-Set-Theory-Infinity/Lebesgue - Measure and Integration 1902|Lebesgue]] later integrates it by observing that the rationals have measure zero, obtaining integral $0$.

The exact function that breaks Riemann’s framework therefore points directly to the idea needed to repair it: **measure how large the exceptional set is.**

### 6. The integral arose inside the Fourier problem, not as an isolated calculus cleanup

Riemann’s dissertation is titled after trigonometric series. Its first sections reconstruct the long dispute among d’Alembert, Euler, Daniel Bernoulli, Lagrange, Fourier, Cauchy, and Dirichlet over arbitrary functions and vibrating strings.

The integration section is not a detour. Fourier coefficients are integrals, and Fourier convergence concerns increasingly complicated function classes. To expand spectral analysis, Riemann first had to specify which functions could enter the coefficient formulas.

This is why the lineage is causal rather than merely chronological:

```text
Fourier broadens the function concept
  → Dirichlet proves convergence under regularity conditions
      → Riemann broadens and tests the integral beneath the coefficients
          → the remaining failures demand measure theory
```

## Historical Context

Riemann wrote the dissertation while preparing to lecture at Göttingen. MacTutor records that he spent roughly thirty months on the Habilitation work and that it supplied what is now called the Riemann integrability condition. The dissertation was completed in 1854, the same year as his famous Habilitation lecture on the foundations of geometry, but it appeared in print only after his death.

His intellectual setting matters:

- **Euler and d’Alembert** had argued over whether arbitrary vibrating-string shapes counted as functions suitable for analysis.
- **Daniel Bernoulli** had proposed decomposition into normal modes.
- **Fourier** made trigonometric representation a general physical method.
- **Cauchy** made limits and convergence explicit but did not finish the arbitrary-function problem.
- **Dirichlet**, Riemann’s strongest mathematical influence, gave the first rigorous broad Fourier convergence theorem.
- **Riemann** changed the question from “is the function smooth enough for familiar calculus?” to “do all fine weighted sums agree?”

The result was a decisive enlargement, but not the final one. Riemann integration remained tied to partitions of the input axis and behaved poorly under some pointwise limits. Nineteenth-century analysis then generated functions with dense discontinuities, exceptional sets, and sequences whose limits escaped the theory.

That pressure led through Jordan and Borel to Lebesgue’s 1902 reconstruction of length and integration. Lebesgue did not make Riemann obsolete. He revealed Riemann integration as a powerful special case inside a broader measure-theoretic framework.

## Lineage

### What came before

```text
Archimedes
  → curved magnitude is trapped by controlled approximation
Newton–Leibniz
  → accumulation and differentiation become inverse operations
Cauchy 1821
  → limits of sums enter rigorous analysis
Fourier 1807–1822
  → arbitrary physical profiles require trigonometric coefficients
Dirichlet 1829
  → Fourier convergence receives explicit regularity conditions
Riemann 1854
  → the integral itself receives a tagged-sum definition and an existence test
```

- [[Fourier - Heat Equation and Trigonometric Representation 1807-1822|Fourier]] made integration over irregular profiles unavoidable.
- [[Cauchy Limits 1821|Cauchy]] supplied the limit-centered program that Riemann sharpened.
- [[Dirichlet - Fourier Series Convergence 1829|Dirichlet]] showed that spectral representation needs hypotheses and exposed the need for a wider theory of integrable functions.

### What Riemann enabled—and what he could not yet do

```text
Riemann 1854
  → discontinuous functions become legitimate objects of integration
  → integrability becomes stability of all fine approximating sums
  → oscillation gives a necessary-and-sufficient interval criterion
  → Fourier coefficients become available for a broader function class
  → but dense persistent oscillation and fragile limit exchange remain
Jordan / Borel / Cantor
  → sets and exceptional points become analyzable objects
Lebesgue 1902
  → size moves from intervals to measurable sets
  → integration becomes compatible with broader limits
Kolmogorov 1933
  → probability becomes normalized measure
Shannon 1948
  → entropy and cross-entropy become expectations
Modern AI
  → expected risk, stochastic approximation, spectral methods, numerical integration
```

The immediate bridge closed by this note is:

**Dirichlet can prove a Fourier series converges for piecewise-regular functions → Riemann defines exactly when discontinuous functions possess the integrals needed for Fourier coefficients → the functions still excluded by interval partitions make Lebesgue measure necessary.**

## Why It Matters for AI

### 1. An average is meaningful only when the approximation procedure is stable

Training often replaces a population objective

$$
R(\theta)=\mathbb E_{Z\sim P}[\ell(\theta;Z)]
$$

with a finite average

$$
\widehat R_n(\theta)=\frac1n\sum_{i=1}^{n}\ell(\theta;Z_i).
$$

A minibatch average is a Monte Carlo estimator, not literally a Riemann sum. But the structural question is the same: under what conditions do finite weighted approximations settle on the intended aggregate rather than depend pathologically on sampling choices?

Riemann’s lesson is methodological: never infer a well-defined limiting objective merely because a finite sum can be computed.

### 2. Numerical quadrature inherits the tagged-partition idea directly

Continuous models repeatedly approximate integrals by finite sums:

- normalization constants and expectations;
- Bayesian evidence and posterior moments;
- continuous-time neural ODE objectives;
- differentiable rendering and volume integration;
- neural operators and PDE solvers;
- attention or convolution viewed as integral operators.

Resolution alone does not guarantee correctness. The function class, local oscillation, sampling rule, and convergence mode decide whether refinement stabilizes the result.

### 3. Nonsmooth does not mean unusable

Modern networks are full of nonsmooth objects: ReLU has a corner, max operations switch branches, decision trees jump, and classification boundaries are discontinuous.

Riemann’s enlargement supplies an early version of a durable principle:

> A finite or even infinite collection of local irregularities can be harmless if it is controlled strongly enough not to dominate the aggregate.

Later measure theory turns this into “almost everywhere” reasoning. That is why gradients can be undefined on a small exceptional set while an expectation or training process remains mathematically workable.

### 4. The failure case teaches why probability must be measure-theoretic

Riemann integration works on intervals through geometric refinement. Data distributions need more:

- discrete atoms and continuous densities in one model;
- probability on high-dimensional spaces;
- singular distributions concentrated on lower-dimensional sets;
- expectations of limits of random functions;
- events that are nonempty but have probability zero.

The path from Riemann’s excluded functions to Lebesgue’s measurable sets is the path from calculus-style averaging to [[../06-Probability/Kolmogorov - Foundations of the Theory of Probability 1933|Kolmogorov probability]]. Modern probabilistic ML lives on the far side of that repair.

### 5. Spectral models still need assumptions about the functions they represent

Fourier features, kernel methods, neural operators, and positional encodings all inherit the old representation question. Coefficients can be computed numerically without proving that truncations converge to the desired function or that integration and limits may be exchanged.

Riemann’s sequence of questions remains the right one:

1. Is the object integrable?
2. Does the representation converge?
3. In what sense does it converge?
4. Which irregularities are harmless, and which change the limit?

### 6. “For every fine approximation” is an adversarial standard

Riemann’s definition does not trust a favorable sequence of partitions. It demands agreement across all sufficiently fine tagged partitions.

That logical shape resembles modern robustness requirements. Good performance under one data split, one seed, one discretization, or one batch order is weaker than stability across an explicitly stated class of perturbations. The historical integral and modern robustness are not the same theorem, but they share a discipline: define the allowed variation, then require the result to survive it.

## The Core Intuition

A discontinuity is not automatically a disaster. The disaster occurs when violent local variation remains spread across enough of the domain that finer sampling never forces different approximations to agree.

Riemann turned integration into a robustness test:

> Subdivide the domain, sample however you like, and add value times width. If every sufficiently fine version converges to the same number, the function has a coherent accumulated value.

That definition carried analysis beyond continuous curves. Its failures then showed exactly what the next idea had to do: stop measuring irregularity only interval by interval and begin measuring the sets on which it occurs.

## References

- Riemann, B. (1854/1867). *Ueber die Darstellbarkeit einer Function durch eine trigonometrische Reihe*. Local primary-source PDF: `corpus/raw/library-papers/riemann-trigonometric-series-integration-1854.pdf`.
- “Riemann integral.” *Encyclopedia of Mathematics*. <https://encyclopediaofmath.org/wiki/Riemann_integral>.
- O’Connor, J. J., and Robertson, E. F. “Bernhard Riemann.” *MacTutor History of Mathematics Archive*. <https://mathshistory.st-andrews.ac.uk/Biographies/Riemann/>.
- [[Dirichlet - Fourier Series Convergence 1829]] — the immediate Fourier-convergence predecessor.
- [[../08-Set-Theory-Infinity/Lebesgue - Measure and Integration 1902]] — the measure-theoretic repair and successor.

## Status

- [x] Primary text downloaded: `corpus/raw/library-papers/riemann-trigonometric-series-integration-1854.pdf`
- [x] Verified with `file`: PDF document, version 1.4
- [x] Primary-source argument checked: tagged sums, arbitrary tags and subdivisions, oscillation condition, and functions with infinitely many discontinuities
- [x] 1854 composition / 1867 publication distinction documented
- [x] Note created: August 19, 2026
- [x] Closes the Dirichlet → Riemann → Lebesgue integration gap
- [x] No orphan: linked to Cauchy, Fourier, Dirichlet, Lebesgue, Kolmogorov, the MOC, and LINEAGE.md

---

*Dirichlet asked when Fourier synthesis tells the truth. Riemann asked when the integral used to build that synthesis exists at all.*
