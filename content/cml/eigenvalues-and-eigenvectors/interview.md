# Interview notes: Eigenvalues and eigenvectors

## Q1. Explain eigenvalues and eigenvectors, and how PCA uses them.

**60-second answer.** An eigenvector of a matrix is a direction the matrix only scales. $A\mathbf{v} =
\lambda\mathbf{v}$, and the eigenvalue $\lambda$ is the scaling factor. For a covariance matrix,
which is symmetric and positive semi-definite, the eigenvectors are orthogonal and point along the
principal axes of the data. Each eigenvalue is the variance along its axis. PCA is the
eigendecomposition of the covariance matrix: keep the top $k$ eigenvectors, and the fraction
$\lambda_i / \sum_j \lambda_j$ tells you how much variance each one explains. In practice I often use
the SVD of the centred data instead, because it avoids forming the covariance matrix and is more stable.

**Follow-ups.**
- *Why are the eigenvectors orthogonal?* Because the matrix is symmetric. Eigenvectors of distinct
  eigenvalues of a symmetric matrix are orthogonal.
- *Does the sign of a principal component matter?* No. Only the line does, so flipping the sign of a
  component changes nothing about the subspace it spans.

## Q2. How would you get the top eigenvector of a huge matrix?

**60-second answer.** Power iteration. Start from a random vector, repeatedly multiply by the matrix
and normalise. The component along the dominant eigenvector grows relative to the others at rate
$|\lambda_2/\lambda_1|$ per step, so the direction converges. Each step is one matrix-vector product,
which is cheap for sparse matrices, and nothing is ever factored. The method needs a clear gap
between the top two eigenvalues. For more than one eigenvector, or a poor gap, use Lanczos or a
randomised SVD. PageRank is power iteration on the link matrix, and it converges quickly because the
damping factor controls the second eigenvalue.

**Follow-ups.**
- *What could make it fail?* Equal top magnitudes, a start vector with no component along the answer,
  or a small gap. A random start handles the second one with probability 1.
- *How do you know it has converged?* Check the residual $\lVert A\mathbf{v} - \lambda\mathbf{v}\rVert$,
  not the change in $\lambda$.

## Q3. What do the eigenvalues of a Hessian tell you?

**60-second answer.** At a critical point, the Hessian's eigenvalues give the curvature in each
principal direction. All positive means a local minimum, all negative a local maximum, and mixed signs
a saddle point. Their ratio is the condition number of the local quadratic, which sets how
gradient descent behaves. A large spread means the loss is shaped like a narrow valley, so a single
learning rate will be too large in some directions and too small in others. That is the usual reason
for momentum or adaptive methods.

**Follow-ups.**
- *Why does a saddle matter in deep learning?* Most critical points in high dimensions are saddles,
  so escaping them matters more than avoiding local minima.
- *Cost of computing the largest eigenvalue?* Power iteration with Hessian-vector products, without
  forming the Hessian.

## Common mistakes

- Calling a general eigen-solver on a symmetric matrix, then getting complex values that are only
  rounding noise.
- Comparing eigenvectors element by element, instead of by $|\cos|$, and reporting a false mismatch
  because of a sign flip.
- Returning the largest algebraic eigenvalue when the question asks for the dominant one by magnitude.
- Stopping power iteration on a fixed number of steps, without checking convergence.
- Using an all-ones start vector without checking that it has a component along the answer.
