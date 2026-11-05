## Hint 1: concept

The minimiser has to satisfy two sets of conditions at once: the constraints $A\mathbf{x} = \mathbf{b}$
and stationarity of the Lagrangian in $\mathbf{x}$. Put both into one linear system in the unknowns
$\mathbf{x}$ and $\boldsymbol\lambda$.

## Hint 2: formula / approach

With $\mathcal{L} = f - \boldsymbol\lambda^\top(A\mathbf{x} - \mathbf{b})$, stationarity in $\mathbf{x}$ is
$Q\mathbf{x} + \mathbf{c} - A^\top\boldsymbol\lambda = 0$. The two sets of conditions form the block
system with matrix $\begin{pmatrix} Q & -A^\top \\ A & 0 \end{pmatrix}$ acting on
$(\mathbf{x}, \boldsymbol\lambda)$, with right-hand side $(-\mathbf{c}, \mathbf{b})$. Check the inputs before
solving: $Q$ must be symmetric and positive definite, and $A$ must have full row rank.

## Hint 3: code shape

Convert each input with `np.array(..., dtype=np.float64)`, which also copies it. Check `.ndim`,
`.shape` and finiteness. Then check that $Q$ is symmetric to a relative tolerance, that
`np.linalg.cholesky(Q)` succeeds, and that `np.linalg.matrix_rank(A)` equals the number of rows.
Build the matrix with `np.block` and the right-hand side with `np.concatenate`, call
`np.linalg.solve`, and split the solution: the first $n$ entries are $\mathbf{x}$, the last $m$ are
$\boldsymbol\lambda$.
