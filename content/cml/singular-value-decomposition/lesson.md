---
title: Singular value decomposition
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Every real matrix $A \in \mathbb{R}^{m \times n}$ factors as

$$A = U \Sigma V^\top, \qquad U^\top U = I, \quad V^\top V = I, \quad \Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_r), \quad \sigma_1 \ge \dots \ge \sigma_r \ge 0,$$

where $r = \min(m, n)$ in the economy form. The numbers $\sigma_i$ are the **singular values**. The columns $u_i$ of $U$ (in
$\mathbb{R}^m$) and $v_i$ of $V$ (in $\mathbb{R}^n$) are the left and right singular vectors. Equivalently,

$$A v_i = \sigma_i u_i, \qquad A^\top u_i = \sigma_i v_i.$$

Read $A = U \Sigma V^\top$ as rotate by $V^\top$, stretch axis $i$ by $\sigma_i$, then rotate by $U$.

Two relations link the SVD to what you know:

- $A^\top A = V \Sigma^2 V^\top$. So the $\sigma_i^2$ are the eigenvalues of $A^\top A$ (forming it squares $\kappa$), and the $v_i$ are its eigenvectors.
- The rank of $A$ is the number of non-zero $\sigma_i$.

The **rank-$k$ truncation** keeps the first $k$ terms: $A_k = \sum_{i \le k} \sigma_i u_i v_i^\top$.

## Worked example

Take $A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$. It is symmetric positive definite, so $U = V$ up to sign.

1. **Gram matrix.** $A^\top A = \begin{pmatrix} 10 & 6 \\ 6 & 10 \end{pmatrix}$. Its eigenvalues are $16$ and $4$, so $\sigma_1 = 4$ and $\sigma_2 = 2$.
2. **Vectors.** $v_1 = (1,1)/\sqrt{2}$ and $v_2 = (1,-1)/\sqrt{2}$. Then $u_1 = A v_1 / \sigma_1 = (1,1)/\sqrt{2}$ and $u_2 = A v_2 / \sigma_2 = (1,-1)/\sqrt{2}$.
   Check: $A v_2 = (2, -2)/\sqrt{2} = 2 u_2$.
3. **Rank one.** $A_1 = 4\, u_1 v_1^\top = \begin{pmatrix} 2 & 2 \\ 2 & 2 \end{pmatrix}$. The residual $A - A_1 = \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}$ has Frobenius norm $2 = \sigma_2$.
4. **Pseudo-inverse and condition number.** $A^+ = V\, \operatorname{diag}(1/4, 1/2)\, U^\top = \begin{pmatrix} 3/8 & -1/8 \\ -1/8 & 3/8 \end{pmatrix}$, which is $A^{-1}$ because $A$ is invertible. Also $\kappa(A) = \sigma_1 / \sigma_2 = 2$.

## Why it works

Write $A = \sum_i \sigma_i u_i v_i^\top$. The $u_i$ are orthonormal and so are the $v_i$, so the cross terms vanish when you
expand a squared norm. That gives two identities. First, $\lVert A \rVert_F^2 = \sum_i \sigma_i^2$. Second, for the
truncation,

$$\lVert A - A_k \rVert_F^2 = \Big\lVert \sum_{i > k} \sigma_i u_i v_i^\top \Big\rVert_F^2 = \sum_{i > k} \sigma_i^2 .$$

Eckart and Young (1936) proved that no matrix of rank $k$ does better than $A_k$ in the Frobenius norm
[library:Mathematics/07-Multivariate-Statistics/Eckart-Young - Best Low-Rank Approximation 1936.md]. Penrose (1955,
Theorem 1) characterises $A^+$ by four conditions; the Library note gives $A^+b$ as the minimum-norm
least-squares solution. Golub and Reinsch (1970, §§1.2–1.3) give the stable SVD route, building on Golub and Kahan (1965)
[library:Mathematics/10-Numerical-Linear-Algebra/Golub-Kahan - Numerical SVD and Pseudoinverse 1965.md].
$\Sigma^+$ inverts each singular value above a cutoff and sets the others to zero.

Three quantities follow from the singular values:

- **Numerical rank.** NumPy's `matrix_rank` counts the singular values above $\max(m, n) \cdot \varepsilon \cdot \sigma_1$.
- **Condition number.** $\kappa = \sigma_{\max}/\sigma_{\min}$ bounds how much a solve amplifies input error.
- **Low-rank parameters.** A LoRA update $\Delta W = BA$, with $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and
  $B \in \mathbb{R}^{d_{\text{out}} \times r}$, has $r(d_{\text{in}} + d_{\text{out}})$ parameters: $65{,}536$ for a $4096 \times 4096$ layer at $r = 8$, or $1/256$ of the full matrix (arithmetic ours). Hu et al. 2021, §4.1, define the update [corpus:library-refresh/01-Mathematics/05-Linear-Algebra/2021 -- Hu et al -- LoRA - Low-Rank Adaptation of Large Language Models -- arXiv -- v2.pdf.md p.4]. It parameterises an update, not the base weights.

## Signal vs. noise

- **Signal: the SVD works on any rectangular matrix.** Eigendecomposition needs a square one. The
  singular values are sorted, so "keep $k$" means "keep the largest".
- **Signal: for $0<k<\operatorname{rank}(A)$, $A_k$ is unique exactly when $\sigma_k > \sigma_{k+1}$.**
  Its optimal error is unique in every case.
- **Noise: the sign of a pair $(u_i, v_i)$, and the algorithm inside the library.** Flipping both leaves
  $u_i v_i^\top$ unchanged. In practice, use a QR- or SVD-based least-squares solver rather than forming $A^+$.

## Where it breaks

- **Normal equations on Longley.** Add an intercept to the raw Longley regressors and the design matrix
  has condition number about $4.86\times10^9$. Solving through $A^\top A$ changes pseudo-inverse entries
  by up to $5.46\times10^{-7}$ relative to a direct SVD; forming the Gram matrix squares the condition
  number and loses precision. Use the SVD cutoff directly.
- **Noise, bias and roundoff.** A pseudo-inverse does not suppress noise: it inverts every singular value
  above the cutoff. Take $A = \operatorname{diag}(1, 10^{-6})$ and $\mathbf{b} = A(1,1)^\top + (0, 10^{-8})^\top$.
  With the default cutoff ($10^{-15}\sigma_1$), $A^+\mathbf{b} = (1, 1.01)$, an error of $0.01$. Truncating at
  $10^{-5}\sigma_1$ returns $(1, 0)$, an error of $-1$: bias instead. A ridge factor $\sigma/(\sigma^2+\lambda)$ with
  $\lambda = 10^{-12}$ gives $5\times10^{5}$ instead of $10^{6}$, an error of $-0.495$ (my reproduction). The
  cutoff and $\lambda$ are modelling choices; roundoff is separate. A computed $\sigma_i$ carries absolute error
  of order $\varepsilon\sigma_1$, which is why cutoffs are relative to $\sigma_1$.
- **Rank-deficient inputs.** $A = (1,2)^\top(1,2)$ has $\sigma = 5$ and a zero singular value, which is dropped,
  so $A^+ = A/25$.
- **Repeated singular values.** The singular vectors are then not unique; any orthonormal basis of the repeated
  eigenspace works.
- **Cost.** A dense SVD costs on the order of $mn\min(m, n)$, and Eckart–Young does not make it affordable.
  Randomised methods can bound an expected range error (Halko et al. 2011, Theorem 10.5, per the Library
  note), which is not a guarantee for one draw, so check the residual.

## Interview angle

The 60-second answer: $U \Sigma V^\top$ is rotate, stretch, rotate. Keeping the top $k$ terms gives the best rank-$k$
matrix in the Frobenius norm. A rank-deficient system needs the pseudo-inverse, which inverts only the singular values
above a relative cutoff. Full answers are in `interview.md`.

## Connections

- **Prerequisites:** `cml/principal-component-analysis`, `cml/eigenvalues-and-eigenvectors`, and
  `cml/matrix-inverse-and-rank`.
- **Unlocks:** `cml/low-rank-approximation` and `cml/matrix-completion-and-factorization`. LoRA learns its low-rank
  factors rather than computing a truncated SVD.
- **Lineage:** Eckart and Young (1936), Penrose (1955), Golub and Kahan (1965), and Golub and Reinsch (1970),
  all linked above.
