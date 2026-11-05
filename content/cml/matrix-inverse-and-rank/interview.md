# Interview notes: Matrix inverse and rank

## Q1. How do you solve Ax = b, and do you compute the inverse?

**60-second answer.** I don't form the inverse. I factor A once with LU and partial pivoting, which
costs $O(n^3)$, and then solve by forward and back substitution, which costs $O(n^2)$ per
right-hand side. The factorisation is reused for every new b. An explicit inverse costs about the
same, is less accurate, and tempts you to multiply by it. For rectangular or overdetermined systems
I use QR or `lstsq`, because they minimise the residual without squaring the condition number the
way $X^\top X$ does.

**Follow-ups.**
- *When is an inverse legitimate?* For a small matrix, or when the inverse itself is the quantity
  you need, such as a covariance inverse for Mahalanobis distance. Even then, compute it through a
  solve.
- *What would make the solve unreliable?* A large condition number. Check it before trusting the
  answer.

## Q2. A regression design matrix is rank deficient. What happens, and what do you do?

**60-second answer.** Rank deficiency means some column is a combination of others, so
$X^\top X$ is singular. The normal equation then has infinitely many solutions, and the
coefficients are not identifiable. The usual cause is collinear features or a dummy-variable trap
(one-hot columns plus an intercept). I would drop a redundant column or level. If I need to keep
all the features, I would regularise with ridge, which adds $\lambda I$ and makes the system
invertible, or use a pseudo-inverse that returns the minimum-norm solution.

**Follow-ups.**
- *How would you detect it?* Compute the rank, or inspect the singular values for values near zero.
  Pivots near zero during elimination also signal it.
- *Does it hurt predictions?* Not necessarily. The fitted values can stay stable even when the
  individual coefficients are not interpretable.

## Q3. What does a near-zero pivot tell you?

**60-second answer.** Either the matrix is truly rank deficient, or it is ill-conditioned, meaning
invertible but close to a singular matrix. Both can give large errors, because the relative error in
the solution can be amplified by up to $\kappa(A)$ relative to the input error. Partial pivoting bounds
the multipliers, not the growth of the entries, and it cannot rescue a matrix whose condition number is
already large. The fix is to
reformulate the problem, scale the features, or regularise. A pivot threshold decides rejection, not
accuracy, so a matrix that passes it can still need that fix.

**Follow-ups.**
- *Why a tolerance rather than an exact zero?* Rounding almost never gives an exact zero, so rank is
  always a numerical rank at some threshold.
- *Which tool gives a principled threshold?* The singular values, from the SVD.

## Common mistakes

- Computing `inv(A) @ b` when `solve(A, b)` was meant. It is slower and less accurate.
- Treating a tiny but nonzero pivot as a valid pivot, and then trusting the output.
- Using an absolute threshold such as `1e-12` on a matrix whose entries are of order $10^{12}$, or
  the reverse. A relative threshold is the right default.
- Forgetting that $X^\top X$ squares the condition number, which is why `lstsq` or QR is preferred.
- Assuming that a matrix from a formula is full rank. Duplicated features are common in practice.
