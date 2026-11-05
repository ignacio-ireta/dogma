---
title: Eigenvalues and eigenvectors
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A square matrix $A$ usually changes both the direction and the length of a vector. An
**eigenvector** is a direction that $A$ only stretches or flips:

$$A\mathbf{v} = \lambda\mathbf{v}, \qquad \mathbf{v} \ne \mathbf{0}.$$

The scalar $\lambda$ is the **eigenvalue**. It can be negative, which means the direction is
reversed. The eigenvalues are the roots of the characteristic polynomial $\det(A - \lambda I) = 0$,
which has degree $n$. For each $\lambda$, the eigenvectors are the null space of $A - \lambda I$.

Matrix type decides how well-behaved the answer is:

| Matrix type | Eigenvalues | Eigenvectors | Typical use |
|---|---|---|---|
| General square | may be complex | may not span $\mathbb{R}^n$ (defective) | dynamics, Markov chains |
| Real symmetric | always real | an orthonormal basis always exists | covariance, Hessians, graph Laplacians |
| Symmetric positive semi-definite | real and $\ge 0$ | orthonormal | covariance and Gram matrices |

**Spectral theorem.** A real symmetric $A$ factors as $A = Q\Lambda Q^\top$, where the columns of
$Q$ are orthonormal eigenvectors and $\Lambda$ is diagonal with the real eigenvalues. So $A$ works in
three steps: rotate into the eigenbasis, stretch each axis by its eigenvalue, and rotate back.

The **dominant eigenpair** is the eigenvalue with the largest absolute value, and its eigenvector.

## Worked example

$A = \begin{pmatrix}2 & 1\\1 & 2\end{pmatrix}$.

1. **Eigenvalues.** $\det(A - \lambda I) = (2-\lambda)^2 - 1 = 0$, so $\lambda = 3$ or $\lambda = 1$.
2. **Eigenvectors.** For $\lambda = 3$, $(A - 3I)\mathbf{v} = 0$ gives $\mathbf{v} \propto (1,1)$. For
   $\lambda = 1$, $\mathbf{v} \propto (1,-1)$. Normalised, these are $\tfrac{1}{\sqrt2}(1,1)$ and
   $\tfrac{1}{\sqrt2}(1,-1)$. They are orthogonal, as the spectral theorem promises.
3. **Check.** $A(1,1) = (3,3) = 3\cdot(1,1)$.

Now power iteration, starting from $\mathbf{v}_0 = (1,0)$. Each step does $\mathbf{v}_{k+1} =
A\mathbf{v}_k / \lVert A\mathbf{v}_k\rVert$. The eigenvalue estimate is the Rayleigh quotient
$\mathbf{v}^\top A\mathbf{v}$ for a unit $\mathbf{v}$.

| $k$ | $\mathbf{v}_k$ | Rayleigh quotient | error in $\lambda$ |
|---|---|---|---|
| 0 | $(1.000,\ 0.000)$ | 2.000 | 1.000 |
| 1 | $(0.894,\ 0.447)$ | 2.800 | 0.200 |
| 2 | $(0.781,\ 0.625)$ | 2.976 | 0.024 |
| 3 | $(0.733,\ 0.681)$ | 2.997 | 0.003 |

The eigenvalue error falls by about 9 per step, and the vector error by about 3.

## Why it works

Write the start vector in the eigenbasis: $\mathbf{v}_0 = c_1\mathbf{q}_1 + c_2\mathbf{q}_2$. After $k$
steps, $A^k\mathbf{v}_0 = c_1\lambda_1^k\mathbf{q}_1 + c_2\lambda_2^k\mathbf{q}_2$. Dividing by
$\lambda_1^k$ gives $c_1\mathbf{q}_1 + c_2(\lambda_2/\lambda_1)^k\mathbf{q}_2$. The direction therefore
approaches $\mathbf{q}_1$ at rate $|\lambda_2/\lambda_1|$ per step. For the worked example that
rate is $1/3$. The two conditions are: $c_1 \ne 0$ (the start has some component along the answer),
and $|\lambda_1| > |\lambda_2|$ (a strict gap).

The eigenvalue error falls at the square of that rate. The Rayleigh quotient is stationary at an
eigenvector, so a vector error $\varepsilon$ produces only an $O(\varepsilon^2)$ error in $\lambda$.

## Signal vs. noise

- **Signal:** symmetric means real eigenvalues and orthogonal eigenvectors. Use a symmetric solver
  for symmetric matrices, never a general one.
- **Signal:** an eigenvector is defined only up to sign, since $A(-\mathbf{v}) = \lambda(-\mathbf{v})$.
  Compare directions with $|\cos|$, never element by element.
- **Signal:** trace is the sum of eigenvalues and determinant their product, from the determinants
  topic.
- **Noise for now:** Jordan normal form, the internals of the QR algorithm, and complex eigenvalues of
  rotations. In practice you call a library.

## Where it breaks

- **Equal magnitudes.** $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ has eigenvalues $+1$ and $-1$. Power
  iteration swaps between two directions and never converges.
- **Start orthogonal to the answer.** On $\begin{pmatrix}2&-1\\-1&2\end{pmatrix}$, the all-ones start is
  an eigenvector for the smaller eigenvalue $1$, so the iteration stalls at $1$ instead of $3$. A
  random start has a nonzero component with probability 1.
- **Small gap.** $|\lambda_2/\lambda_1| = 0.99$ needs thousands of steps. Shift the matrix, or use a
  Krylov method such as Lanczos.
- **Stopping rule.** The change in $\lambda$ can be small while the vector is still off by about
  $\sqrt{\text{tol}}$, because of the quadratic relation above. Stop on the residual $\lVert A\mathbf{v} -
  \lambda\mathbf{v}\rVert$ instead.

## Interview angle

"Explain eigenvalues and eigenvectors, and how PCA uses them." Answer shape: an eigenvector is a
direction the matrix only scales. For a covariance matrix, which is symmetric and positive
semi-definite, the eigenvectors are the principal axes and each eigenvalue is the variance along its
axis. PCA is the eigendecomposition of the covariance, and the fraction $\lambda_i/\sum_j\lambda_j$ is
the variance each component explains. Follow-ups: why the eigenvectors are orthogonal (symmetry), and
how to find only the top eigenvector of a huge sparse matrix (power iteration or Lanczos). Full answers
are in `interview.md`.

## Connections

- **Prerequisites:** `cml/matrix-inverse-and-rank` (null spaces), `math/determinants-and-trace` (the
  characteristic polynomial), `cml/vector-and-matrix-norms` (unit vectors).
- **Unlocks:** `cml/principal-component-analysis`, `cml/singular-value-decomposition`,
  `cml/hessians-definiteness-and-critical-points` (Hessian eigenvalues set the curvature that
  gradient descent meets), `gnn/random-walks-and-pagerank` (power iteration on a Markov matrix),
  `gnn/graph-laplacian-eigenvectors`.
- **Lineage:** von Mises and Pollaczek-Geiringer described the power method in 1929. The QR algorithm,
  which computes all eigenvalues of a general matrix, followed in 1961. Brin and Page's 1998 PageRank
  paper ranked web pages by running power iteration on the link matrix.
