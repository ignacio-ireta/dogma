---
title: Vector operations
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A vector $\mathbf{a} \in \mathbb{R}^n$ is an ordered list of $n$ numbers: a point, or an arrow
from the origin. Five operations cover almost everything ML does with vectors:

| Operation | Formula | Shape in → out | Meaning |
|---|---|---|---|
| Linear combination | $\alpha\mathbf{a} + \beta\mathbf{b}$ | $(n),(n) \to (n)$ | Stretch and add arrows |
| Dot product | $\mathbf{a}\cdot\mathbf{b} = \sum_i a_i b_i$ | $(n),(n) \to$ scalar | How much the arrows agree |
| Norm (L2) | $\lVert\mathbf{a}\rVert = \sqrt{\mathbf{a}\cdot\mathbf{a}}$ | $(n) \to$ scalar | Length |
| Angle | $\cos\theta = \dfrac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}$ | $(n),(n) \to [-1,1]$ | Direction agreement, scale removed |
| Projection | $\operatorname{proj}_{\mathbf{b}}(\mathbf{a}) = \dfrac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}\,\mathbf{b}$ | $(n),(n) \to (n)$ | The shadow of $\mathbf{a}$ on the line through $\mathbf{b}$ |

Decision rule for the sign of $\mathbf{a}\cdot\mathbf{b}$: **positive** means they point roughly
the same way ($\theta < 90°$), **zero** means orthogonal, **negative** means roughly opposite.

The projection splits $\mathbf{a}$ into two parts:

$$\mathbf{a} = \underbrace{\operatorname{proj}_{\mathbf{b}}(\mathbf{a})}_{\text{along } \mathbf{b}} + \underbrace{\big(\mathbf{a} - \operatorname{proj}_{\mathbf{b}}(\mathbf{a})\big)}_{\text{residual, } \perp\, \mathbf{b}}$$

## Worked example

$\mathbf{a} = (1, 2, 3)$, $\mathbf{b} = (1, 1, 1)$.

1. $\mathbf{a}\cdot\mathbf{b} = 1 + 2 + 3 = 6$
2. $\mathbf{b}\cdot\mathbf{b} = 3$
3. Coefficient $= 6/3 = 2$, so $\operatorname{proj}_{\mathbf{b}}(\mathbf{a}) = (2, 2, 2)$. This is the
   **mean of $\mathbf{a}$** repeated: projecting onto the all-ones vector gives the average.
4. Residual $= (1,2,3) - (2,2,2) = (-1, 0, 1)$. Check: $(-1,0,1)\cdot(1,1,1) = 0$, so it's
   orthogonal. The residual is the data *centred* on its mean.

In NumPy: `a @ b` is the dot product, and `np.sqrt(a @ a)` is the norm.

## Why it works

Write the projection as $c\,\mathbf{b}$ and require the residual to be orthogonal to $\mathbf{b}$
(Strang, *Introduction to Linear Algebra*, §4.2, derives the same formula):

$$(\mathbf{a} - c\,\mathbf{b})\cdot\mathbf{b} = 0 \;\Rightarrow\; c = \frac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}.$$

That orthogonality condition also makes $c\,\mathbf{b}$ the point on the line closest to
$\mathbf{a}$ (Pythagoras: any other point adds a perpendicular component to the error). **Least
squares is this same idea**, with the line through $\mathbf{b}$ replaced by the column space of a
matrix $X$ (see `cml/normal-equation`).

## Signal vs. noise

- **Signal:** dot product ↔ agreement, the projection formula, "residual ⟂ direction", cosine
  as scale-free similarity. Interviews assume these with no hesitation.
- **Signal:** shapes. Most bugs in ML code are shape bugs: `(n,)` vs `(n,1)` vs `(1,n)`.
- **Noise for now:** abstract inner-product spaces, complex vectors, cross products (3-D only,
  rarely used in ML).

## Where it breaks

- **Zero vector:** $\mathbf{b} = \mathbf{0}$ makes the projection undefined. Raise an error,
  don't divide. Don't detect it with $\mathbf{b}\cdot\mathbf{b} = 0$: for a tiny nonzero
  $\mathbf{b}$ that product underflows to 0, and for a huge one it overflows. Check the largest
  entry instead, then divide $\mathbf{b}$ by it. The projection is unchanged, because it depends
  only on the line through $\mathbf{b}$.
- **Scale dominates raw dot products:** a long vector has large dot products with everything.
  That's why retrieval uses cosine similarity or normalises embeddings first.
- **Integer arrays:** `np.array([1, 2])` is an integer array. Convert with `dtype=float` before
  dividing or writing into the array.
- **Overflow:** squaring an entry above about $1.34 \times 10^{154}$ overflows float64. Even
  `np.linalg.norm([3e200, 4e200])` returns `inf` (and `[3e-200, 4e-200]` underflows to `0.0`).
  The fix is to divide by the largest absolute entry first, then multiply back. `np.hypot` and
  BLAS `nrm2` do this. See `cml/vector-and-matrix-norms`.

## Interview angle

"Explain cosine similarity and when you'd prefer it over Euclidean distance." Answer shape:
cosine compares *direction*, Euclidean compares *position*. For embeddings, direction carries
the meaning and the norm often reflects frequency or length, so cosine (or normalising and then
taking the dot product, which is equivalent and faster) is the default. Follow-ups: what happens
with the zero vector, and why normalised dot product equals cosine.

## Connections

- **Unlocks:** `cml/cosine-similarity`, `cml/vector-and-matrix-norms`, `cml/matrix-multiplication`
  (every entry of $AB$ is a dot product), `cml/orthogonal-projection`, `cml/normal-equation`,
  `dl/scaled-dot-product-attention` (attention scores are dot products).
- **Lineage:** Grassmann's *Ausdehnungslehre* (1844) introduced the algebra of vectors and inner
  products. The projection-as-best-approximation idea runs through Gauss's least squares (1809).
