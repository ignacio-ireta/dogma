# Finite Precision, Rounding Error and Stable Computation


## Source reference

**David Goldberg, “What Every Computer Scientist Should Know About Floating-Point Arithmetic,” _ACM Computing Surveys_ 23(1), 5–48, March 1991.** DOI: [10.1145/103162.103163](https://doi.org/10.1145/103162.103163).

Full text (not imported): 44-page scan of the published article. The title page identifies Goldberg, Xerox PARC, volume 23, number 1, March 1991, and journal pages 5–48.

Goldberg’s paper is the primary source for the exposition below. The modern-AI sections are interpretation and application, not claims that the paper anticipated today’s architectures.

## The problem that made this idea necessary

Analysis works with real numbers, but a computer has finitely many bit patterns. Most real numbers therefore have no exact machine representation. Even (0.1) has an infinite repeating expansion in binary. Arithmetic must map an exact result back into a finite set, and that rounding happens repeatedly inside a computation.

This creates a problem exact algebra does not answer: **when a mathematically correct formula is executed with finite precision, which digits—if any—remain trustworthy?**

The question has two parts:

1. **Representation:** how far is a stored number from the real number it represents?
2. **Algorithm:** how do many individually small rounding errors interact with a computation’s structure?

A theorem can be exact while its obvious implementation is unreliable. Conversely, a computed answer can be accurate even though every intermediate operation rounded. Numerical analysis connects exact mathematics to finite machines.

## Key ideas

### 1. Floating point is a finite, uneven grid

Ignoring special values, a normalized floating-point number has the form

[
pm(d_0.d_1ldots d_{p-1})_eta,eta^e,
]

with base (eta), precision (p), and a bounded exponent (e). The grid is approximately uniform relative to scale, not in absolute distance: adjacent numbers near (2^{20}) are much farther apart than adjacent numbers near (2^{-20}).

Goldberg uses two complementary measures: an **ulp** (“unit in the last place”) measures error in units of the local spacing; **relative error**, (|widehat{x}-x|/|x|), measures it in proportion to the exact value. If rounding chooses the nearest representable number, the error is at most one half ulp. For ordinary normalized numbers this is often modeled as

[
operatorname{fl}(x)=x(1+delta), qquad |delta|le u,
]

where (u) is unit roundoff (Goldberg uses “machine epsilon” for the corresponding bound). The model needs qualifications near zero, overflow, underflow, and exceptional values.

### 2. Small local error does not imply small final error

Suppose two already-rounded quantities are almost equal. Subtracting removes their shared leading digits and exposes uncertain low-order digits. Goldberg distinguishes **benign cancellation**, where exact inputs are subtracted with well-guarded arithmetic, from **catastrophic cancellation**, where earlier approximation errors become large relative to a small difference.

His example uses (b=3.34), (a=1.22), and (c=2.28). In three-digit decimal arithmetic, (b^2) rounds to (11.2), while (4ac) rounds to (11.1). Their computed difference is (0.1), although the exact difference is (0.0292). The subtraction did not create all that error; it revealed error already present in the products.

Algebraically equivalent formulas therefore need not be computationally equivalent. A quadratic root can lose its accurate digits when (-b) and (sqrt{b^2-4ac}) nearly cancel; rationalizing the numerator changes the numerical path without changing the mathematical root. Likewise, (x^2-y^2) may be safer as ((x-y)(x+y)) when (x) and (y) are close.

### 3. Guard digits and correct rounding control each operation

A guard digit retains extra information while aligning operands for addition or subtraction. Goldberg’s Theorem 2 shows that one guard digit bounds the relative rounding error of subtracting two floating-point inputs by less than (2u). This cannot restore information lost earlier, but prevents the subtraction hardware from adding avoidable large error.

IEEE arithmetic goes further: basic operations are specified *as if* the exact real result were computed and then rounded once to the destination format. This is **correct rounding**. It supplies a precise contract and makes basic results reproducible across conforming implementations.

Round-to-nearest needs a tie rule. “Round ties to even” avoids the systematic upward drift produced by always rounding halfway cases upward. Directed modes—toward zero, (+infty), and (-infty)—support interval bounds and specialized numerical reasoning.

### 4. IEEE special values make the system operational

IEEE 754 standardized infinities, signed zero, NaNs, gradual underflow, rounding modes, and exception flags alongside finite numbers.

- Infinities let overflow and limiting operations propagate without an immediate trap.
- **NaN** marks an invalid numeric result and propagates through later arithmetic.
- **Signed zero** preserves directional information in limits and some reciprocal identities.
- **Subnormal numbers** fill the gap around zero, so underflow loses precision gradually rather than jumping from the smallest normal number to zero.
- **Flags** record invalid operations, division by zero, overflow, underflow, and inexact results while allowing execution to continue.

Goldberg explains and synthesizes this standard; he did not invent it. His examples include IBM’s 1968 retrofit of a guard digit to System/360 double precision and the then-recent adoption of IEEE arithmetic.

### 5. Stability belongs to an algorithm; conditioning belongs to a problem

A problem is **well-conditioned** when small input changes produce small changes in its exact answer. An algorithm is **stable** when its computed answer is the exact answer to a nearby problem, or has another controlled error appropriate to the problem.

No algorithm can manufacture information absent from the data. A backward-stable solver can still have a large forward error on an ill-conditioned system. This boundary is already visible in [Least Squares, Pseudoinverses and Numerical Stability](1955-1970%20--%20Penrose%2C%20Golub%20and%20Reinsch%20--%20Least%20Squares%2C%20Pseudoinverses%20and%20Numerical%20Stability%20--%20Concept%20Note.md): QR or SVD can avoid gratuitously squaring a condition number, but cannot make nearly dependent features independent.

## Historical context

Electronic computers changed the error problem’s scale. Arithmetic was repeated millions of times, hardware formats differed, and a program could move between machines with different rounding and exceptional behavior. Numerical analysts developed error bounds and stable algorithms; hardware designers developed formats; IEEE 754, first issued in 1985, supplied a common operational standard.

Goldberg’s 1991 survey arrived after the standard, when floating point was ubiquitous but poorly understood outside numerical analysis. Its contribution was a bridge: representation, rounding proofs, cancellation examples, IEEE semantics, compiler behavior, and operating-system exception handling appear in one argument. The development was not inevitable. Decimal and binary formats, alternative number systems, different exception policies, and different rounding rules were possible; IEEE 754 made one set of tradeoffs precise and portable.

The paper also corrects a persistent misconception: floating-point errors are not random dust handled by “adding a tolerance.” Error depends on scale, operation order, conditioning, and formula choice. Goldberg notes that fixed-tolerance “equality” is not even a transitive relation.

## What came before → what it enabled

```text
Real-number analysis and exact algebra
  → finite positional arithmetic and rounding-error analysis
  → guard digits and stable reformulations
  → IEEE 754 (1985): specified operations and exceptional values
  → Goldberg (1991): unified mathematical and systems account
  → dependable numerical libraries and mixed-precision analysis
  → modern AI training and inference on finite accelerators
```

Cauchy’s limits supplied a language of approximation and bounds. Matrix theory supplied exact decompositions; Golub–Kahan and Golub–Reinsch supplied algorithms designed for floating-point machines. IEEE 754 supplied an arithmetic contract; Goldberg explained why representation, algorithm choice, language semantics, and hardware behavior must be analyzed together.

This branch joins [stochastic gradients](../08-Optimization/2018%20--%20Bottou%2C%20Curtis%20and%20Nocedal%20--%20Stochastic%20Gradients%2C%20Noise%20and%20Computation%20--%20Study%20Note.md): sampling noise and rounding error are different, even when both perturb an update. One may average away under assumptions; the other can be biased, amplified by cancellation, or correlated with execution order.

## Why it matters for AI

Modern AI is not executed over (mathbb R). It runs in binary32, binary16, bfloat16, tensor-float variants, integers, and mixed-precision accumulators.

### Stable objectives and normalizations

Softmax is invariant to subtracting a common constant. Computing

[
p_i=rac{e^{z_i-m}}{sum_j e^{z_j-m}},qquad m=max_j z_j,
]

keeps the largest exponent at (e^0=1) and reduces overflow. Log-sum-exp uses the same reformulation. This is the principle behind Goldberg’s examples: preserve the mathematical function while changing the path through finite arithmetic.

### Dot products, reductions, and attention

Matrix multiplication is built from sums of products. Floating-point addition is not associative, so different reduction trees, kernels, or parallel schedules can produce different low bits. Pairwise or compensated summation, wider accumulators, and careful kernel design control this error. Attention logits, exponentials, normalization sums, and value-weighted reductions inherit these concerns.

### Mixed precision needs an error budget

Lower precision saves memory bandwidth and increases accelerator throughput, but narrows exponent range or significand precision. Loss scaling protects small gradients from underflow; wider accumulators protect long dot products; master weights preserve updates too small to change a low-precision parameter directly. These techniques place precision where the computation is sensitive to it.

### Reproducibility and quantization

NaNs and infinities can identify invalid operations or overflow only if the system exposes where they first arose. Nondeterministic reduction order can change low bits and, through a long optimization trajectory, eventually produce a different model. Reproducibility claims must specify arithmetic, kernels, operation order, and seeds—not only equations and data.

Quantization deliberately maps parameters or activations to a smaller finite set. The same questions return: What is the grid? Is rounding biased? Where does saturation occur? Which transformations are well-conditioned? Quantization is not IEEE floating point, but belongs to the same lineage of finite representation plus controlled error.

## A small calculation

Exact arithmetic in Goldberg’s discriminant example gives

[
b^2-4ac=11.1556-11.1264=0.0292.
]

Rounding each product to three significant decimal digits first gives (11.2-11.1=0.1). The absolute error is (0.0708); the relative error is about (2.42), or 242%. Each product was locally close to correct, yet the final small difference was not. This is the shortest useful rebuttal to “rounding errors are always tiny.”

## Primary-source locators

- pp. 5–7: motivation, finite representation, and formats.
- pp. 7–8: ulps, relative error, and the rounding bound.
- pp. 8–12: guard digits, cancellation, and stable reformulations.
- pp. 12–14: correctly rounded operations and round-to-even.
- pp. 14–27: IEEE formats, special quantities, gradual underflow, exceptions, and rounding modes.
- pp. 27–34: flags, instruction sets, languages, compilers, and exception handling.
- pp. 35–48: proofs, conversion, summation analysis, summary, and references.

## Boundaries

- Goldberg is a tutorial and synthesis. IEEE 754, guard digits, and numerical error analysis predate it.
- A correctly rounded primitive does not guarantee that a whole algorithm is stable.
- Stability does not repair an ill-conditioned problem.
- More precision can postpone failure without fixing a structurally unstable formula.
- The AI applications above are later interpretations, not evidence of direct historical influence.
