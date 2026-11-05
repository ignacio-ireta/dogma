# Interview notes: Vector operations

## Q1. What does the dot product measure, geometrically?

**60-second answer.** It measures how much two vectors agree in direction, scaled by their
lengths: $\mathbf{a}\cdot\mathbf{b} = \lVert\mathbf{a}\rVert\lVert\mathbf{b}\rVert\cos\theta$.
Positive means they point roughly the same way, zero means orthogonal, negative means opposite.
Dividing out the lengths gives cosine similarity, a scale-free measure of direction. In ML it's
everywhere: a linear model's score is a dot product between weights and features, every entry of
a matrix product is one, and attention scores are dot products between queries and keys.

**Follow-ups.**
- *Why scale by $1/\sqrt{d}$ in attention?* Dot products of random $d$-dimensional vectors have
  variance growing with $d$. Scaling keeps the softmax out of its saturated region.
- *Cost?* $O(n)$ multiply-adds, memory-bound for large $n$.

## Q2. How do you project a vector onto another, and why does it matter?

**60-second answer.** $\operatorname{proj}_{\mathbf{b}}(\mathbf{a}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\mathbf{b}\cdot\mathbf{b}}\mathbf{b}$.
It's the closest point to $\mathbf{a}$ on the line through $\mathbf{b}$, and the residual is
orthogonal to $\mathbf{b}$. It matters because least-squares regression is exactly this
projection onto the column space of the design matrix, and PCA projects onto the directions of
maximum variance.

**Follow-ups.**
- *Edge case?* $\mathbf{b} = \mathbf{0}$ makes it undefined. Validate before dividing.
- *Onto a subspace?* $P = X(X^\top X)^{-1}X^\top$. In practice, use QR or `lstsq`, never an
  explicit inverse.

## Q3. Cosine similarity vs. Euclidean distance for embeddings?

**60-second answer.** Cosine compares direction and ignores magnitude. Euclidean compares
position, so magnitude matters. For text embeddings the norm often tracks length or frequency
rather than meaning, so cosine is the default. If vectors are L2-normalised, the two give the
same ranking, because $\lVert\mathbf{a}-\mathbf{b}\rVert^2 = 2 - 2\cos\theta$. That's why vector
databases normalise once and then use the faster dot product.

## Common mistakes

- Dividing by $\mathbf{b}\cdot\mathbf{b}$ without checking for zero.
- Integer arrays silently truncating results when written into in place.
- Confusing `(n,)`, `(n,1)` and `(1,n)` shapes, so a broadcast gives an $n \times n$ matrix
  instead of a vector.
- Using `*` (elementwise) where `@` (dot or matrix product) was meant.
