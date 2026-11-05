---
title: Cosine similarity
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

For nonzero vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^d$, cosine similarity is the angle
between them with length removed:

$$\cos\theta(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert} \in [-1, 1].$$

Three measures answer three different questions. Pick by what the magnitude means:

| Measure | Formula | Range | Sees | Ignores | Use when |
|---|---|---|---|---|---|
| Dot product | $\mathbf{a}\cdot\mathbf{b}$ | unbounded | direction and length | nothing | magnitude is signal (e.g. popularity-weighted scores) |
| Cosine | $\mathbf{a}\cdot\mathbf{b} / (\lVert\mathbf{a}\rVert\lVert\mathbf{b}\rVert)$ | $[-1,1]$ | direction | length | magnitude is noise (text length, embedding norm) |
| Euclidean distance | $\lVert\mathbf{a}-\mathbf{b}\rVert$ | $[0,\infty)$ | position | nothing | absolute position matters (sensor readings, k-means) |

Three rules follow:

1. **Cosine is a dot product of unit vectors.** With $\hat{\mathbf{a}} = \mathbf{a}/\lVert\mathbf{a}\rVert$,
   $\cos\theta = \hat{\mathbf{a}}\cdot\hat{\mathbf{b}}$. Normalise once, and every comparison is one
   dot product.
2. **For unit vectors, Euclidean and cosine rank the same.**
   $\lVert\hat{\mathbf{a}}-\hat{\mathbf{b}}\rVert^2 = 2 - 2\cos\theta$.
3. **All pairs at once.** For $A \in \mathbb{R}^{n\times d}$ and $B \in \mathbb{R}^{m\times d}$ with
   row-normalised copies $\hat A$ and $\hat B$, the matrix $S = \hat A\,\hat B^\top \in \mathbb{R}^{n\times m}$
   holds $S_{ij} = \cos(A_i, B_j)$. One matrix multiply.

Zero vectors have no direction, so cosine is undefined for them. This repo follows scikit-learn:
any pair involving a zero-norm row scores $0.0$. The convention is a choice, so state it in the
code and the docstring.

## Worked example

$\mathbf{a} = (1, 0)$, $\mathbf{b} = (1, 1)$, $\mathbf{c} = (0, 2)$.

1. $\mathbf{a}\cdot\mathbf{b} = 1$, $\lVert\mathbf{a}\rVert = 1$, $\lVert\mathbf{b}\rVert = \sqrt{2}$, so
   $\cos\theta_{ab} = 1/\sqrt{2} \approx 0.707$.
2. $\mathbf{a}\cdot\mathbf{c} = 0$, so $\cos\theta_{ac} = 0$. Orthogonal, even though $\lVert\mathbf{c}\rVert = 2$.
3. $\mathbf{b}\cdot\mathbf{c} = 2$, $\lVert\mathbf{b}\rVert\lVert\mathbf{c}\rVert = 2\sqrt{2}$, so
   $\cos\theta_{bc} = 1/\sqrt{2}$.
4. Scale check: $10\mathbf{b} = (10, 10)$ gives $\cos = 10/(10\sqrt{2}) = 1/\sqrt{2}$. Unchanged.

The dot product tells a different story: $\mathbf{a}\cdot\mathbf{b} = 1$ but $(10\mathbf{b})\cdot\mathbf{a} = 10$.
Same direction, different score.

## Why it works

Expanding $\lVert\mathbf{a}-\mathbf{b}\rVert^2 = (\mathbf{a}-\mathbf{b})\cdot(\mathbf{a}-\mathbf{b})$ gives

$$\mathbf{a}\cdot\mathbf{b} = \tfrac12\big(\lVert\mathbf{a}\rVert^2 + \lVert\mathbf{b}\rVert^2 - \lVert\mathbf{a}-\mathbf{b}\rVert^2\big).$$

This is the law of cosines in vector form. Dividing by $\lVert\mathbf{a}\rVert\lVert\mathbf{b}\rVert$ and
using unit vectors gives the identity in rule 2. Cauchy–Schwarz ($|\mathbf{a}\cdot\mathbf{b}| \le
\lVert\mathbf{a}\rVert\lVert\mathbf{b}\rVert$) keeps the ratio in $[-1,1]$. Floating point can still
push a value to $1.0000000000000002$, so the implementation clips.

Centring is the one change that alters the answer. The cosine of mean-centred vectors is the
Pearson correlation coefficient. Same algebra, different centre point.

## Signal vs. noise

- **Signal:** normalise, then dot. The three-way choice in the table. Cosine lies in $[-1,1]$.
  Pearson correlation is centred cosine.
- **Signal:** the zero-vector case is a product decision, not a math fact. Name the convention.
- **Noise for now:** Jaccard, Mahalanobis and learned metrics. Know they exist; you rarely need
  them in a first pass.

## Where it breaks

- **Magnitude carries meaning.** A user with 1 purchase and a user with 100 purchases in the same
  proportions look identical under cosine. Keep the norm as a feature, or use the dot product.
- **Non-negative features compress the range.** Counts and TF-IDF have no negative entries, so
  every cosine lies in $[0,1]$ and everything looks somewhat similar. Centring spreads the range.
- **A shared mean direction.** Some embedding spaces have vectors that all point roughly the
  same way, so typical cosines are high. Subtract the mean embedding, then look at the
  distribution of cosines on random pairs before trusting a threshold.
- **Overflow and underflow.** Squaring entries near $10^{154}$ overflows float64, and squaring
  entries near $10^{-162}$ underflows to zero. Rescale each row by its largest entry before the
  norm. `np.linalg.norm` does not do this for the 2-norm (it computes $\sqrt{\mathbf{x}\cdot\mathbf{x}}$
  directly), so it overflows on the same inputs.

## Interview angle

"Explain cosine similarity and when you'd prefer it over Euclidean distance." Answer shape:
cosine compares direction and ignores length, while Euclidean compares position. Use cosine when
length is a nuisance, as with text and embeddings. Use Euclidean when scale is meaningful. If the
vectors are normalised, the two rank identically, so vector stores normalise once and then use the
cheaper dot product. Follow-ups: the zero vector, and when cosine misleads. Full answers are in
`interview.md`.

## Connections

- **Prerequisites:** `cml/vector-operations` (dot product, norm, angle) and
  `cml/vector-and-matrix-norms`.
- **Unlocks:** `cml/covariance-and-correlation` (correlation is centred cosine),
  `llm/text-embedding-models-from-decoders` and `llm/retrieval-augmented-generation-pipeline`
  (ranking by cosine), `dl/scaled-dot-product-attention` (attention scores are raw dot products;
  cosine is the same score with the lengths removed).
- **Lineage:** the vector space model of information retrieval, developed in the 1960s and 1970s,
  represented documents as term vectors and ranked them by cosine similarity. The same ranking is
  still the default in dense retrieval.
