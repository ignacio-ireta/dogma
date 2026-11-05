# Karl Pearson — On Lines and Planes of Closest Fit (1901)

#person #paper #milestone

**Person:** Karl Pearson (1857–1936)
**Work:** *On Lines and Planes of Closest Fit to Systems of Points in Space* (1901)
**Primary source:** *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, Series 6, 2(11), 559–572
**Reference trail:** Stanford STATS 202 PCA notes, stats.org.uk PCA historical summary, and later historical commentary on PCA's prehistory

## Title
PCA begins as a geometry problem.

Before "dimensionality reduction" became a phrase, Pearson asked a simpler, more beautiful question:

> if a cloud of points lives in high-dimensional space, what line or plane fits it best?

That is the birth of principal component analysis.

Not yet with modern matrix notation. Not yet with scree plots and sklearn. But the core move is already here:
- center the data,
- find the direction that captures the most structure,
- project onto a lower-dimensional subspace,
- keep as much variation as possible while throwing away redundancy.

This is one of the deepest bridges in the whole Library, because it joins **geometry, statistics, and compression** in a single move.

## Paper Reference
**Pearson, K. (1901).** *On Lines and Planes of Closest Fit to Systems of Points in Space.* *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, 2(11), 559–572.

Historical note: Pearson introduced the method in 1901. Harold Hotelling's 1933 work later generalized and popularized the language of **principal components**.

## Key Ideas

### 1. Best fit is no longer just for curves — it is for subspaces
Pearson takes a set of multivariate observations and asks for the **line** or **plane** that lies closest to them in aggregate.

That sounds innocent, but it changes the scale of statistics. Instead of fitting one response variable from one predictor, he is fitting an entire **lower-dimensional geometry** to a data cloud.

This is the moment multivariate structure becomes something you can summarize systematically.

### 2. "Closest fit" means minimizing squared reconstruction error
Pearson's criterion is geometric:
- project the points onto a line or plane,
- measure the squared distances from the original points to that subspace,
- choose the subspace that makes the total error as small as possible.

This is the ancestor of a huge amount of modern ML thinking:
- compress while losing as little information as possible,
- represent data in fewer dimensions,
- optimize an objective defined by squared error.

So PCA is not just visualization. It is one of the earliest clean statements of **lossy compression with mathematical guarantees**.

### 3. The variance view and the distance view are secretly the same story
Modern PCA is usually taught this way:
- the **first principal component** is the direction of maximum variance,
- the second is the next orthogonal direction of maximum remaining variance,
- and so on.

Pearson's formulation emphasizes closest fit in Euclidean distance, while later expositions emphasize variance maximization.

These are two faces of the same object.

That equivalence matters because it reveals a pattern that shows up everywhere later:
**geometry and optimization keep turning out to be the same language in disguise.**

### 4. Orthogonality becomes a way to separate information
Once you have one best direction, the next useful direction should not simply repeat it.
So PCA insists on orthogonal axes.

This gives you:
- uncorrelated component scores,
- ordered directions of importance,
- a clean decomposition of structure.

This is why PCA became such a durable workhorse. It turns messy covariance into an ordered basis.

### 5. The matrix future was already waiting inside the idea
Pearson did not have the full modern SVD presentation, but the modern machinery lands here naturally:
- center the data matrix `X`,
- compute eigenvectors of `X^T X` or the covariance matrix,
- project data onto those eigenvectors,
- recover Eckart–Young low-rank approximation as the best rank-k compression.

So Pearson's geometric note becomes the ancestor of:
- Hotelling's multivariate statistics,
- eigenvalue methods,
- PCA in data analysis,
- SVD-based compression,
- latent semantic methods,
- embedding visualization,
- and representation learning intuition more broadly.

## Lineage

```text
Descartes 1637 (coordinates)
  └── Cayley 1858 (matrices as objects)
      └── Gauss 1809 (least squares + squared-error fitting)
          └── Pearson 1901 (best-fit subspaces / PCA)
              ├── Hotelling 1933 (principal components formalized)
              ├── Eckart–Young 1936 (best low-rank approximation)
              ├── multivariate statistics
              ├── data compression and denoising
              └── modern representation learning
```

More specifically:
- **Built on:** [[../03-Analytic-Geometry/Descartes - Analytic Geometry 1637|Descartes]] for the coordinate worldview, [[../02-Algebraic-Revolution/Cayley - Theory of Matrices 1858|Cayley]] for the matrix language that later clarifies PCA, and [[../06-Probability/Gauss - Least Squares and Normal Errors 1809|Gauss]] for the squared-error logic of best fit
- **Leads to:** Hotelling's multivariate methods, factor analysis, covariance geometry, and [[../../AI-ML/02-Linear-Algebra/Eckart-Young Best Low-Rank Approximation 1936|Eckart–Young 1936]] on best low-rank approximation
- **Connected to:** modern SVD, embeddings, visualization, denoising, feature engineering, and covariance-based reasoning in statistics and ML

## Why It Matters

### For mathematics
This is the point where geometry stops being about shapes on paper and becomes about **structure in data**.

Pearson turns a cloud of observations into something you can analyze with lines, planes, orthogonality, and error minimization.
That is a major shift.

It is also the moment linear algebra becomes emotionally legible:
not just symbols, but a way of asking what directions in the data actually matter.

### For AI/ML
A lot of modern ML is still Pearson with more layers on top.

PCA shows up when you want to:
- reduce dimensionality before modeling,
- denoise signals,
- visualize embeddings,
- compress information,
- inspect covariance structure,
- or explain latent dimensions in human language.

And conceptually, it trains the same instinct that later powers autoencoders and learned representations:
**keep the signal, discard the redundancy.**

## The Real Historical Role
Pearson did not write PCA in the modern textbook style, and Hotelling later systematized it more fully.
But Pearson is where the original spark lives.

He looked at a messy system of points and asked for the best lower-dimensional truth hiding inside it.
That question is still alive in modern ML.

## Leads To
- [[../_MOC|Mathematics Lineage]]
- [[../02-Algebraic-Revolution/Cayley - Theory of Matrices 1858|Cayley - Theory of Matrices 1858]]
- [[../06-Probability/Gauss - Least Squares and Normal Errors 1809|Gauss - Least Squares and Normal Errors 1809]]
- [[../../AI-ML/02-Linear-Algebra/Eckart-Young Best Low-Rank Approximation 1936|Eckart–Young 1936]]
- Future note: Hotelling 1933 on principal components as a full multivariate framework
- Future note: spectral theorem / eigenvalue decomposition as the algebraic spine of PCA

## References
- Pearson, K. (1901). *On Lines and Planes of Closest Fit to Systems of Points in Space.* *Philosophical Magazine*, Series 6, 2(11), 559–572.
- Stanford STATS 202, *Principal Components Analysis* notes, used for the modern optimization/eigendecomposition framing and the Pearson/Hotelling historical markers.
- stats.org.uk PCA historical summary, used as a concise secondary source for lineage and terminology.

## Status
- [x] Note created (April 29, 2026)
- [ ] Add Hotelling (1933) as the next explicit multivariate-statistics note
- [ ] Add a future note on spectral decomposition / eigenvalue theory to complete the Cayley → Pearson → Eckart–Young chain

---

*Pearson's move was simple and enormous: if data are a cloud, then intelligence begins by finding its axes.*