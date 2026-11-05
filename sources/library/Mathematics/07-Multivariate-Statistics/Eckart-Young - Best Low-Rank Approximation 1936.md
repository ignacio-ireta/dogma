# Carl Eckart and Gale Young - Best Low-Rank Approximation (1936)

#person #paper #milestone

**People:** Carl Eckart (1902-1973) and Gale Young (1912-1990)  
**Work:** *The Approximation of One Matrix by Another of Lower Rank* (1936)  
**Primary source:** *Psychometrika*, 1(3), 211-218  
**Paper:** `corpus/raw/library-papers/eckart-young-low-rank-1936.pdf`  
**Source note:** The local article PDF was extracted from the Internet Archive scan of *Psychometrika* 1(3), September 1936. The DOI record is `10.1007/BF02288367`; Springer currently exposes the article page reliably, while the direct PDF route returned an HTML access page during this build.

## Title
Compression becomes a theorem.

Pearson and Hotelling had already made principal components intelligible:

- Pearson: what line or plane lies closest to a data cloud?
- Hotelling: how do we rotate correlated variables into ordered, uncorrelated components?

But a sharper question was still missing:

> if we are allowed to keep only `r` dimensions, which rank-`r` matrix is closest to the original matrix?

That is the Eckart-Young problem.

It matters because PCA is not only a coordinate rotation. It is a controlled loss of information. Eckart and Young give the mathematical permission slip: keep the largest canonical components, discard the smaller ones, and the resulting lower-rank matrix is the least-squares best approximation.

That is the bridge from PCA as a statistical method to SVD as the workhorse of compression, denoising, matrix factorization, latent semantic analysis, recommender systems, and modern representation learning.

## Paper Reference
**Eckart, C., & Young, G. (1936).** *The Approximation of One Matrix by Another of Lower Rank.* *Psychometrika*, 1(3), 211-218.

Local source:
- `corpus/raw/library-papers/eckart-young-low-rank-1936.pdf`
- Full issue preserved as `corpus/raw/library-papers/eckart-young-low-rank-1936-issue.pdf`

Reference trail:
- DOI article page: <https://doi.org/10.1007/BF02288367>
- Springer article page: <https://link.springer.com/article/10.1007/BF02288367>
- Internet Archive issue scan: <https://archive.org/details/sim_psychometrika_1936-09_1_3>

## What Problem Made This Necessary?
Early factor analysis assumed that many test scores could be explained by a smaller number of underlying factors.

That assumption sounds simple, but it hides a hard mathematical constraint:

- the original score table is a matrix,
- the simplified model must have lower rank,
- the entries of that lower-rank matrix are not independent,
- and ordinary least-squares normal equations cannot be written down directly.

So the problem becomes:

> among all matrices of rank `r`, which one is closest to the given matrix?

Hotelling had given PCA a statistical identity. Eckart and Young give the optimal approximation theorem beneath it.

Without this step, dimensionality reduction is a useful recipe. With it, truncation becomes an optimal operation under squared-error distance.

## Key Ideas

### 1. Matrices get a geometry of their own
Eckart and Young define a scalar product for matrices by summing products of corresponding entries.

That makes a matrix behave like a point in a high-dimensional Euclidean space:

- matrix length becomes the square root of summed squared entries,
- distance between matrices becomes reconstruction error,
- approximation becomes a geometric least-squares problem.

This is the Frobenius-norm viewpoint before the modern term dominates the story.

### 2. Rank is a complexity constraint
The approximating matrix must have rank `r`, smaller than the rank of the original matrix.

That rank constraint is the entire point. It says:

we are not just fitting entries; we are limiting the number of independent directions allowed in the representation.

In factor analysis language, this means fewer latent factors. In modern ML language, it means lower-dimensional representation, compression, and controlled model capacity.

### 3. The canonical form is the solution path
The paper uses what we now recognize as a singular-value-style decomposition: a real matrix can be transformed by orthogonal matrices into a diagonal rectangular form with nonnegative diagonal entries.

Eckart and Young call these diagonal entries canonical multipliers.

Modern language:

\[
A = U \Sigma V^T
\]

where the diagonal entries of \(\Sigma\) are singular values.

The old notation is different, but the move is familiar: rotate the row space and column space until the matrix is only a list of independent strengths.

### 4. Best rank-r approximation means keep the largest components
Once the matrix is in canonical form, the theorem becomes almost brutal:

keep the largest `r` diagonal components and set the rest to zero.

Equivalently, in modern SVD notation:

\[
A_r = \sum_{i=1}^{r} \sigma_i u_i v_i^T
\]

This \(A_r\) is the closest rank-`r` approximation to \(A\) under summed squared entry error.

The error is the discarded energy:

\[
\|A - A_r\|_F^2 = \sum_{i=r+1}^{R} \sigma_i^2
\]

So the theorem does more than say "truncate." It tells you exactly what truncation costs.

### 5. The solution is usually unique
The best approximation is unique unless there is a tie at the cutoff among the retained and discarded canonical multipliers.

That is an important practical detail. If \(\sigma_r = \sigma_{r+1}\), the boundary between "signal kept" and "signal discarded" is not uniquely determined.

Modern interpretation: when the spectrum has a clean gap, the low-rank representation is stable and meaningful. When the spectrum is flat near the cutoff, the chosen subspace is less trustworthy.

## Historical Context
The paper appears in the first volume of *Psychometrika*, inside the world of factor analysis, test batteries, and the attempt to explain many observed scores using fewer latent dimensions.

That origin matters. This theorem did not enter history as an abstract machine-learning trick. It entered because psychologists and statisticians were trying to compress score matrices without lying to themselves.

The mathematical difficulty was not the desire to simplify. Everyone wanted that. The difficulty was proving which simplification was best.

Eckart and Young solve that problem by moving into canonical matrix components. The original applied question becomes a pure matrix theorem, and that theorem later escapes its birthplace.

By the time SVD becomes computationally practical, this result is waiting underneath every serious low-rank approximation:

- PCA reconstruction,
- image compression,
- noise reduction,
- latent semantic analysis,
- recommender systems,
- low-rank model compression,
- matrix completion,
- and spectral methods throughout ML.

## Lineage

```text
Descartes 1637 (coordinates)
  └── Cayley 1858 (matrices as algebraic objects)
      └── Cauchy 1829 (spectral theorem for symmetric matrices)
          └── Pearson 1901 (best-fit subspaces for data clouds)
              └── Hotelling 1933 (principal components as ordered variables)
                  └── Eckart-Young 1936 (best rank-r matrix approximation)
                      ├── numerical SVD pipelines
                      ├── latent semantic analysis
                      ├── recommender-system matrix factorization
                      ├── low-rank compression and denoising
                      └── modern representation learning
```

More specifically:

- **Built on:** [[../02-Algebraic-Revolution/Cayley - Theory of Matrices 1858|Cayley]] for matrices as objects, [[../02-Algebraic-Revolution/Cauchy - Spectral Theorem for Symmetric Matrices 1829|Cauchy]] for eigen-structure, [[Pearson - Principal Components 1901|Pearson]] for closest-fit subspaces, and [[Hotelling - Principal Components 1933|Hotelling]] for principal components as a statistical method.
- **Solved the next problem:** PCA and factor analysis needed a theorem saying which lower-rank reconstruction is actually best.
- **Leads to:** SVD-based PCA, the Eckart-Young-Mirsky theorem, low-rank approximation in numerical linear algebra, latent semantic indexing, recommender systems, and model compression.

## Why It Matters for AI
Modern AI is full of enormous matrices pretending not to be enormous.

Eckart-Young is one of the reasons we know how to simplify them without hand-waving:

- **Embeddings:** large vector spaces often have dominant directions and redundant dimensions.
- **Recommendation systems:** user-item matrices can be approximated by latent factors.
- **Latent semantic analysis:** word-document matrices can be compressed into semantic axes.
- **Denoising:** small singular values often carry noise more than signal.
- **Model compression:** large weight matrices can sometimes be approximated by lower-rank factors.
- **Representation learning:** the linear case gives the cleanest theorem for the compression instinct that neural networks later generalize.

The AI lesson is not "always use low rank." The lesson is sharper:

when a matrix's spectrum decays, structure concentrates. Keeping the strongest components gives a principled compressed representation, and the discarded singular values tell you what you paid.

That is a very adult idea. Useful, but not magical. Exactly how we like it.

## The Real Historical Role
Eckart and Young turn dimensionality reduction from a geometric/statistical habit into an optimal approximation theorem.

Pearson finds the subspace.
Hotelling orders the components.
Eckart and Young prove what happens when you reconstruct with only the strongest ones.

That is the chain:

geometry -> statistics -> optimal compression.

Modern AI keeps walking that same path, only with larger matrices and more expensive GPUs. Same mountain, louder boots.

## Leads To
- [[../_MOC|Mathematics Lineage]]
- [[Pearson - Principal Components 1901]]
- [[Hotelling - Principal Components 1933]]
- [[../02-Algebraic-Revolution/Cauchy - Spectral Theorem for Symmetric Matrices 1829]]
- Future note: Mirsky 1960 on unitarily invariant norms
- Future note: Golub-Reinsch 1970 as the computational SVD bridge
- Future AI/ML note: latent semantic analysis or recommender-system matrix factorization

## References
- Eckart, C., & Young, G. (1936). *The Approximation of One Matrix by Another of Lower Rank.* *Psychometrika*, 1(3), 211-218. Local PDF: `corpus/raw/library-papers/eckart-young-low-rank-1936.pdf`.
- Springer DOI page for bibliographic confirmation: <https://link.springer.com/article/10.1007/BF02288367>.
- Internet Archive scan of *Psychometrika* 1(3), September 1936, used for the source PDF: <https://archive.org/details/sim_psychometrika_1936-09_1_3>.
- Richard M. Johnson (1963), *On a theorem stated by Eckart and Young*, used as a historical reminder that the 1936 theorem became central in lower-rank approximation and factor analysis.

## Status
- [x] Note created (June 19, 2026)
- [x] Source PDF verified: `corpus/raw/library-papers/eckart-young-low-rank-1936.pdf`
- [x] Closes the Hotelling 1933 -> SVD/low-rank approximation gap
- [ ] Add Mirsky 1960 later if the lineage needs the norm-generalization step

---

*Eckart-Young is the moment "throw away the small dimensions" becomes mathematics instead of vibes.*
