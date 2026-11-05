## Hint 1 — concept

The loss is convex, so a zero gradient identifies a global minimiser. With a positive L2 penalty and both classes present, that minimiser is unique; at zero penalty, rank deficiency can leave several equivalent fits. The intercept
is part of the same vector as the coefficients but is not part of the penalty. Write the penalty as a
diagonal that is zero in the intercept's position.

## Hint 2 — formula / approach

Let $A = [\mathbf{1}, X]$ and $\theta = (b, \mathbf{w})$, so $\eta = A\theta$. Then
$\nabla J = \frac1n A^\top(\sigma(\eta) - y) + \Lambda\theta$ and
$H = \frac1n A^\top \mathrm{diag}(p(1-p))A + \Lambda$, where $\Lambda = \mathrm{diag}(0, \ell_2, \dots, \ell_2)$.
Newton's step is $\Delta = -H^{-1}\nabla J$. Compute $\log(1 + e^\eta)$ as $\max(\eta, 0) + \log_1p(e^{-|\eta|})$.

## Hint 3 — code shape

Validate first: `X` 2-D and non-empty, `y` 1-D with entries in {0, 1} and both classes present, `l2` finite and
non-negative, `max_iter` an integer at least 1, `tol` positive. Loop up to `max_iter` times: compute the
gradient, return `(theta[0], theta[1:])` when `np.max(np.abs(grad)) <= tol`, otherwise solve for the step with
`np.linalg.solve(H, -grad)` and update `theta`. Add a backtracking loop on the objective if a plain step overshoots.
After the loop, check the gradient once more and raise `RuntimeError` if it still fails.
