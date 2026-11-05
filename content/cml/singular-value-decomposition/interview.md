# Interview notes — Singular value decomposition

## Q1. What is the SVD, and what does it give you that the eigendecomposition does not?
**60-second answer.** Every real matrix factors as $A = U\Sigma V^\top$: a rotation, a stretch by the singular values, and
another rotation. The singular values are the square roots of the eigenvalues of $A^\top A$. Unlike an eigendecomposition,
the SVD exists for any rectangular matrix, and its singular values are always real and non-negative. For a symmetric
positive semi-definite matrix, the two coincide.

**Follow-ups.**
- *What is the numerical rank?* The number of singular values above a tolerance tied to $\sigma_{\max}$. NumPy uses
  $\max(m, n)\,\varepsilon\,\sigma_{\max}$.
- *Why not get the singular values from the eigenvalues of $A^\top A$ in practice?* Forming $A^\top A$ squares the condition
  number, so small singular values lose accuracy.

## Q2. What is the best rank-$k$ approximation, and how do you know it is optimal?
**60-second answer.** Keep the first $k$ terms $\sum_{i \le k} \sigma_i u_i v_i^\top$. The low-rank approximation theorem says no rank-$k$ matrix
is closer to $A$ in the Frobenius norm. Orthonormality gives the truncation error
$\sum_{i>k} \sigma_i^2$ under the square root; the low-rank approximation theorem additionally proves that no other rank-at-most-$k$ matrix improves it. The same truncation compresses data, denoises it when the discarded
values are noise, and gives low-rank factors for recommender systems.

**Follow-ups.**
- *How does this relate to LoRA?* A rank-$r$ update $BA$ has $r(d_{\text{in}} + d_{\text{out}})$ parameters, instead of
  $d_{\text{in}} d_{\text{out}}$ for a full matrix. LoRA learns its factors rather than computing a truncated SVD.
- *What if two singular values are equal?* Non-uniqueness occurs when a positive repeated singular value straddles the truncation boundary; repeated values wholly inside or outside it do not cause this.

## Q3. How do you invert a rank-deficient or non-square matrix?
**60-second answer.** Use the pseudo-inverse $A^+ = V\Sigma^+U^\top$. It inverts each singular value above a cutoff and sets
the rest to zero. The cutoff must be relative to $\sigma_{\max}$, so that a matrix scaled down by $10^{-16}$ is not zeroed.
For least squares, $A^+ b$ is the minimum-norm solution. In production I would call a QR- or SVD-based solver rather than
form $A^+$ explicitly.

**Follow-ups.**
- *Why not invert $A^\top A$?* It is singular whenever $A$ is rank-deficient, and it squares the condition number.
- *What is the condition number?* $\sigma_{\max}/\sigma_{\min}$. It bounds how much a solve can amplify error in the input.
- *Why does a pseudo-inverse amplify noise?* It inverts each singular value above the cutoff, so noise along that direction is multiplied by $1/\sigma$. Truncating the direction instead trades that noise for bias.

## Common mistakes
- Using an absolute cutoff for the pseudo-inverse, which zeroes everything in a small-scale matrix.
- Taking the error of a truncation as the largest discarded singular value, rather than the square root of the sum of their squares.
- Using $U$ where $V$ belongs, which only shows up for non-square or non-symmetric matrices.
- Forming $(A^\top A)^{-1}$ for least squares, which fails for rank-deficient data.
- Assuming the singular vectors are unique when singular values repeat.
