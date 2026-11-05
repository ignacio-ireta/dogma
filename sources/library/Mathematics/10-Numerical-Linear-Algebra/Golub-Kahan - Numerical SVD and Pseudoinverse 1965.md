# Golub and Kahan - Numerical SVD and Pseudoinverse (1965)

#person #paper #milestone

**People:** Gene H. Golub (1932-2007) and William Kahan (1933-)  
**Work:** *Calculating the Singular Values and Pseudo-Inverse of a Matrix* (1965)  
**Primary source:** *Journal of the Society for Industrial and Applied Mathematics, Series B: Numerical Analysis*, 2(2), 205-224  
**Paper:** `corpus/raw/library-papers/golub-kahan-svd-pseudoinverse-1965.pdf`  
**Source note:** The local PDF is William Kahan's hosted scan of the 1965 SIAM paper. `file` verifies it as a PDF document. The official DOI record is `10.1137/0702016`.

## Title
Low-rank geometry becomes computable.

Eckart and Young had given the theorem:

> if you want the best rank-`r` approximation of a matrix under squared error, keep the largest singular components.

That is beautiful, but beauty does not multiply matrices on a computer.

By the 1960s, electronic computation had made matrix problems larger, noisier, and less forgiving. It was no longer enough to know that a singular value decomposition existed. Scientists needed a stable way to compute singular values, singular vectors, and the pseudoinverse of rectangular or rank-deficient matrices.

Golub and Kahan supply that missing engineering mathematics. They turn SVD from a canonical form in a theorem into a practical numerical algorithm.

## Paper Reference

**Golub, G. H., & Kahan, W. (1965).** *Calculating the Singular Values and Pseudo-Inverse of a Matrix.* *Journal of the Society for Industrial and Applied Mathematics, Series B: Numerical Analysis*, 2(2), 205-224.

Local source:
- `corpus/raw/library-papers/golub-kahan-svd-pseudoinverse-1965.pdf`

Reference trail:
- Kahan-hosted PDF: <https://people.eecs.berkeley.edu/~wkahan/GlbKhnSv.pdf>
- SIAM DOI page: <https://doi.org/10.1137/0702016>
- Semantic Scholar record: <https://www.semanticscholar.org/paper/Calculating-the-singular-values-and-pseudo-inverse-Golub-Kahan/9238de967f7bd30d5fb5b2a9450e42023add7910>

## What Problem Made This Necessary?

The SVD answers an idealized question:

> how can any matrix be rotated into independent input-output directions with nonnegative strengths?

But a computer asks a harsher version:

> how do we compute those directions without numerical error destroying the answer?

The naive route is tempting: form \(A^T A\), diagonalize it, and take square roots of the eigenvalues. That works on paper, but it can be numerically ugly. Forming \(A^T A\) squares the condition number, magnifies roundoff, and makes small singular values especially vulnerable.

Those small singular values are not decorative. They decide:

- whether a least-squares problem is ill-conditioned,
- whether a matrix is effectively low-rank,
- how much noise should be damped,
- and how the pseudoinverse should behave.

So the missing link after Eckart-Young was not another existence theorem. It was a stable computational path.

## Key Ideas

### 1. SVD is the right diagonalization for arbitrary matrices

Eigenvalue decompositions are naturally square-matrix tools.

The SVD works for rectangular matrices:

\[
A = U \Sigma V^*
\]

where \(U\) and \(V\) are unitary or orthogonal, and \(\Sigma\) is diagonal-shaped with nonnegative singular values.

That matters because data matrices are usually rectangular:

- observations by features,
- users by items,
- documents by words,
- pixels by latent factors,
- examples by embedding dimensions.

The SVD is the diagonalization that fits real data tables.

### 2. Bidiagonalization reduces the problem without wrecking the geometry

Golub and Kahan reduce the original matrix to bidiagonal form using orthogonal transformations.

That is the decisive numerical move.

Orthogonal transformations preserve lengths and angles, so they do not amplify errors the way arbitrary transformations can. The dense rectangular matrix is simplified into a bidiagonal matrix whose singular values can be computed more efficiently and stably.

In lineage terms:

- Cayley made matrices algebraic objects.
- Eckart-Young made singular components optimal for low-rank approximation.
- Golub-Kahan made singular components computable in floating-point arithmetic.

### 3. The pseudoinverse becomes a practical least-squares tool

The Moore-Penrose pseudoinverse generalizes the inverse to matrices that are rectangular, singular, or nearly singular.

Using SVD:

\[
A^+ = V \Sigma^+ U^*
\]

where \(\Sigma^+\) flips the nonzero singular values into reciprocals.

That formula is simple, but also dangerous if implemented blindly. Tiny singular values create huge reciprocals, which amplify noise. The SVD exposes exactly where instability lives, so the pseudoinverse can damp spurious oscillation and cancellation rather than pretending the data are cleaner than they are.

This is the computational ancestor of regularized least squares, truncated SVD, and many inverse-problem methods.

### 4. Numerical linear algebra becomes a field, not just arithmetic

This note belongs after Eckart-Young because it changes the nature of the question.

Before numerical linear algebra, a theorem could say:

> this decomposition exists.

After numerical linear algebra, a theorem also has to answer:

> can we compute it accurately with finite precision?

That finite-precision turn is essential for modern AI. Training, inference, embeddings, PCA, recommender systems, and scientific ML all live inside approximate arithmetic. Stability is not polish. Stability is whether the method means what it claims to mean.

### 5. Bidiagonalization becomes a long-lived algorithmic pattern

The Golub-Kahan bidiagonalization idea does not stay frozen in 1965.

It becomes part of the lineage leading to:

- Golub-Reinsch SVD algorithms,
- Krylov subspace methods,
- large sparse SVD,
- iterative least-squares solvers,
- latent semantic indexing,
- recommender-system factorization,
- and randomized low-rank approximation.

Modern large-scale ML often cannot afford a full dense SVD. But the idea of reaching singular structure through stable projections and reduced subproblems is still everywhere.

## Historical Context

This paper appears at exactly the moment when computation is becoming a mathematical pressure.

Earlier linear algebra could often be written as exact symbolic manipulation. By the mid-20th century, the real problem was different: machines were solving large numerical systems in floating point, and roundoff error was no longer an annoyance. It was part of the theory.

Golub and Kahan are working in the world shaped by Householder transformations, QR methods, pseudoinverses, and the rise of numerical analysis as a serious discipline.

The paper's historical role is not that it invented the abstract SVD. It made the SVD operational.

That distinction matters. Modern AI does not run on existence theorems. It runs on algorithms that survive finite precision, scale, and noise.

## Lineage

```text
Descartes 1637 (coordinates)
  └── Cayley 1858 (matrices as algebraic objects)
      └── Cauchy 1829 (spectral structure for symmetric matrices)
          └── Pearson 1901 (best-fit subspaces for data)
              └── Hotelling 1933 (ordered principal components)
                  └── Eckart-Young 1936 (best rank-r approximation)
                      └── Golub-Kahan 1965 (stable numerical SVD and pseudoinverse)
                          ├── Golub-Reinsch 1970 (standard SVD computation)
                          ├── truncated SVD and latent semantic indexing
                          ├── sparse / iterative SVD methods
                          ├── recommender-system matrix factorization
                          └── low-rank compression in modern ML
```

More specifically:

- **Built on:** [[../07-Multivariate-Statistics/Eckart-Young - Best Low-Rank Approximation 1936|Eckart-Young]] for the optimality of low-rank approximation, [[../02-Algebraic-Revolution/Cayley - Theory of Matrices 1858|Cayley]] for matrix algebra, and [[../02-Algebraic-Revolution/Cauchy - Spectral Theorem for Symmetric Matrices 1829|Cauchy]] for the spectral worldview.
- **Solved the next problem:** SVD and pseudoinverse theory needed a stable floating-point algorithm before they could become routine scientific tools.
- **Leads to:** numerical SVD libraries, truncated SVD, latent semantic analysis, recommender systems, inverse problems, and scalable low-rank approximation.

## Why It Matters for AI

Modern AI is not only mathematics. It is computed mathematics.

Golub-Kahan matters because so many AI objects are too large, rectangular, noisy, or rank-deficient for clean symbolic algebra:

- **PCA at scale:** compute dominant singular directions of centered data.
- **Embeddings:** inspect, compress, and denoise high-dimensional representation matrices.
- **Recommender systems:** factor sparse user-item matrices into latent factors.
- **Latent semantic analysis:** turn term-document matrices into semantic axes.
- **Inverse problems:** recover signals from noisy or underdetermined measurements.
- **Model compression:** approximate large weight matrices with lower-rank factors.
- **Optimization diagnostics:** singular values reveal conditioning and sensitivity.

The deeper lesson is this:

> representation learning needs decomposition, but decomposition needs stability.

Without stable numerical SVD, the beautiful low-rank story stays trapped in small examples. With it, the story becomes a working computational primitive.

## The Real Historical Role

Eckart-Young says what the best low-rank approximation is.

Golub-Kahan says how to get it without the computer lying to you.

That is the hinge:

theorem -> algorithm -> usable ML primitive.

This is also why numerical linear algebra belongs in the lineage to AI. Modern models are built from matrices, but they are trained and analyzed by machines that round, truncate, approximate, and iterate. The bridge from pure structure to reliable computation is not optional. It is the road.

## Leads To

- [[../_MOC|Mathematics Lineage]]
- [[../07-Multivariate-Statistics/Eckart-Young - Best Low-Rank Approximation 1936]]
- Future note: Golub-Reinsch 1970 on practical SVD and least-squares solutions
- Future note: Lanczos / Krylov methods for large sparse eigenvalue and singular-value problems
- Future note: latent semantic indexing as SVD enters information retrieval
- Future AI/ML note: matrix factorization for recommender systems

