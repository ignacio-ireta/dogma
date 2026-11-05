# Leibniz — Differential Substitution and the Chain Rule (1676)

#person #milestone #read #calculus #chain-rule #automatic-differentiation

**Person:** Gottfried Wilhelm Leibniz (1646–1716)  
**Era:** Scientific Revolution  
**Key contribution:** Differentiation becomes compositional. Name an inner expression, differentiate the inner and outer quantities separately, then substitute one differential into the other. This is the operational ancestor of the chain rule, Jacobian products, automatic differentiation, and backpropagation.  
**Paper:** `corpus/raw/library-papers/child-leibniz-chain-rule-1917.pdf` — J. M. Child, “The Manuscripts of Leibniz on His Discovery of the Differential Calculus. Part II (Continued),” *The Monist* 27(3), 1917, pp. 411–454. The article prints English translations of Leibniz’s 1676–1677 manuscripts with historical commentary.

---

## The Problem That Made This Necessary

[[Newton-Leibniz - Fundamental Theorem of Calculus|Newton and Leibniz]] made differentiation a general operation, but a list of derivative formulas was not yet enough.

Analytic geometry produced roots, quotients, implicit relations, and expressions nested inside other expressions. Even if one knew how to differentiate `u²` and `a + bx + cx²`, that did not by itself say how to differentiate

$$
y=\sqrt{a+bx+cx^2}.
$$

Solving every compound expression by a new trick would make calculus problem-specific again. The missing question was:

> If an output changes through an intermediate quantity, how do the two local changes combine?

Leibniz’s decisive move was substitution. Name the inner quantity, find its differential, find the outer differential with respect to it, and compose the results. Complicated change could now be assembled from simple change.

---

## Key Ideas

### 1. Name the intermediate dependency

Suppose

$$
y=f(u), \qquad u=g(x).
$$

Then `y` depends on `x` **through** `u`. The chain rule states

$$
\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx},
$$

or

$$
(f\circ g)'(x)=f'(g(x))g'(x).
$$

The outer derivative is evaluated at the inner function’s output. For the motivating example, let

$$
u=a+bx+cx^2, \qquad y=\sqrt{u}.
$$

Then

$$
\frac{dy}{du}=\frac{1}{2\sqrt{u}},
\qquad
\frac{du}{dx}=b+2cx,
$$

so

$$
\frac{dy}{dx}=\frac{b+2cx}{2\sqrt{a+bx+cx^2}}.
$$

Two local rules plus one composition rule replace a special-purpose derivation.

### 2. Leibniz made substitution part of the calculus

Child’s translation shows the idea forming in Leibniz’s manuscripts during 1676. Discussing compound fractions and irrational expressions, Leibniz says that their differentials are “far better found separately and then substituted” (pp. 421–422). He introduces a letter for the expression under a root, differentiates the named expression, and substitutes that result into the differential of the root.

The intermediate variable exposes a computational graph:

```text
x
└── u = a + bx + cx²
    └── y = √u
```

Each edge carries a local derivative. The derivative of the whole path is their product.

Child records the historical roughness too. Leibniz’s 1676 calculation contains a sign error, and his rules become clearer across the 1676–1677 manuscript sequence. This is an operational discovery in progress, not a polished modern theorem.

### 3. The rule is local linear composition

The modern reason the chain rule works is deeper than symbolic cancellation. Near `x`,

$$
g(x+h)=g(x)+g'(x)h+o(h).
$$

Near `g(x)`,

$$
f(g(x)+k)=f(g(x))+f'(g(x))k+o(k).
$$

Substituting the first approximation into the second gives

$$
f(g(x+h))-f(g(x))=f'(g(x))g'(x)h+o(h).
$$

The best local linear map for a composition is the composition of the two local linear maps.

In many dimensions, if

$$
g:\mathbb{R}^n\to\mathbb{R}^m,
\qquad
f:\mathbb{R}^m\to\mathbb{R}^p,
$$

then

$$
J_{f\circ g}(x)=J_f(g(x))J_g(x).
$$

The scalar chain rule is the one-dimensional case of Jacobian matrix composition.

### 4. Long compositions become products of local sensitivity

For `h_n∘h_{n-1}∘⋯∘h_1`, the derivative is an ordered product of local derivatives. The order matters because Jacobian multiplication is generally not commutative.

This creates two computational directions:

- **Forward mode:** carry an input perturbation forward through Jacobian–vector products.
- **Reverse mode:** carry an output sensitivity backward through vector–Jacobian products.

The mathematics is the same chain rule. The computational difference is how the products are associated so that full Jacobian matrices need not be materialized.

### 5. Differential “cancellation” is a mnemonic, not the proof

Leibniz notation makes `dy/dx = (dy/du)(du/dx)` look as if `du` cancels. That mnemonic is powerful, but the theorem is not justified by treating every differential as an ordinary number. Its rigorous content is composition of derivatives as local linear maps.

The notation nevertheless preserves a real structure: the intermediate dependency disappears from the final input-to-output sensitivity only after its two local effects have been composed.

---

## Historical Context

Seventeenth-century calculus had rules for powers, products, quotients, tangents, and quadratures, but algebraic expressions naturally nest. Radicals contain polynomials; coordinates depend on parameters; implicit curves tie several quantities together.

Leibniz’s 1676 manuscripts show him moving from isolated rules toward a calculus of dependencies. Child identifies the separate letter for the differentiated quantity and substitution as the steps that let Leibniz handle quotients and roots of functions. By July 1677, the algebraic rules were more systematic.

The date should not be flattened into a myth. Leibniz used the chain-rule structure in 1676; he did not state today’s fully quantified theorem in modern function notation. The modern statement emerged gradually as functions and derivatives became explicit objects in Lagrange, Cauchy, and nineteenth-century analysis. [[../04-Analysis/Cauchy Limits 1821|Cauchy’s limits]] and [[../04-Analysis/Weierstrass Epsilon-Delta 1861|Weierstrass’s quantified error control]] later supplied the rigorous setting.

Leibniz supplied the durable operational grammar: **separate a compound dependency, differentiate its parts, and substitute the local changes**.

---

## Lineage

### What came before

```text
Viète and Descartes
  → symbolic equations describe general curves
Fermat
  → local perturbation finds tangents and extrema
Barrow
  → tangent and area problems become reciprocal
Newton & Leibniz
  → differentiation and integration become general operators
Leibniz’s substitution method
  → derivatives of nested dependencies compose
```

The chain rule was necessary because a general derivative without a rule for composition could not remain general once expressions became nested.

### What this enabled

```text
Chain rule
  → derivatives of nested functions
  → Jacobians: multivariable derivatives as linear maps
  → variational and constrained optimization
  → computation graphs and automatic differentiation
  → reverse-mode differentiation
  → backpropagation through deep neural networks
```

It also creates a new numerical problem: repeated derivatives can shrink or grow exponentially along a long chain. In neural networks, this becomes the vanishing- and exploding-gradient problem.

---

## Why It Matters for AI

### 1. A neural network is a nested function

A feed-forward network is a composition:

$$
\hat y=h_L(h_{L-1}(\cdots h_1(x;\theta_1)\cdots;\theta_{L-1});\theta_L).
$$

Training asks how a final scalar loss changes with an early parameter. The chain rule makes that global question local:

$$
\frac{\partial L}{\partial \theta_1}
=
\frac{\partial L}{\partial h_L}
\frac{\partial h_L}{\partial h_{L-1}}
\cdots
\frac{\partial h_1}{\partial \theta_1}.
$$

Every layer only needs its own local derivative.

### 2. Backpropagation is efficient chain-rule bookkeeping

A naive method recomputes overlapping derivative paths. Reverse-mode automatic differentiation starts at the scalar loss and reuses intermediate sensitivities while moving backward through the graph: dynamic programming applied to the chain rule.

For a scalar loss and millions or billions of parameters, reverse mode computes the whole gradient at a cost comparable to a small number of forward evaluations. Deep learning is computationally viable because composition gives gradients reusable structure.

### 3. Gradient pathologies are products along the chain

Repeated Jacobian multiplication explains why depth is difficult:

$$
\left\|\frac{\partial h_L}{\partial h_1}\right\|
\le
\prod_{k=2}^{L}\left\|J_{h_k}\right\|.
$$

If typical factors have norm below one, gradients vanish; above one, they can explode. Activation functions, initialization, normalization, residual connections, gating, and gradient clipping are responses to this product.

### 4. Tensor systems execute Leibniz’s dependency idea

PyTorch, JAX, and similar systems record primitive operations and local derivative rules, then compose those rules over a graph. Leibniz’s “find separately and then substitute” has become executable infrastructure.

```text
symbolic substitution
  → chain rule
  → Jacobian composition
  → reverse-mode automatic differentiation
  → backpropagation
  → trainable deep models
```

Without the chain rule, calculus could optimize a single formula. With it, learning can assign responsibility through billions of nested operations.

---

## Source Notes

- The local source is a verified 45-page PDF of Child’s 1917 *Monist* article. It contains translated Leibniz manuscripts, including the 1676 treatment of a root of a compound expression and the instruction to find differentials separately and substitute them.
- Child’s article is both a secondary historical study and an English presentation of primary manuscript material. The original manuscript is not a modern “chain rule paper,” so this note distinguishes Leibniz’s operational discovery from the later theorem.
- Journal record and DOI: <https://doi.org/10.5840/monist191727324>.
- Open-access scan: <https://archive.org/details/jstor-27900650>.
- Historical cross-check: <https://en.wikipedia.org/wiki/Chain_rule>.
- Broader chronology: <https://mathshistory.st-andrews.ac.uk/HistTopics/The_rise_of_calculus/>.

---

## Status

- [x] Source downloaded and verified as PDF: `corpus/raw/library-papers/child-leibniz-chain-rule-1917.pdf`
- [x] Local note created: September 2, 2026
- [ ] Next hinge: connect Jacobian composition to the first explicit reverse-mode differentiation algorithms and backpropagation.
