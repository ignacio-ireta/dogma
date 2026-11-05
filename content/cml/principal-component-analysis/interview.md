# Interview notes — Principal component analysis

## Q1. What is PCA, and how would you compute it?
**60-second answer.** PCA finds orthogonal directions along which the centred data has the largest variance. Centre
the columns and take the SVD of the centred matrix. The right singular vectors are the principal axes, and each axis's
variance is $s_i^2/(n-1)$. The top $k$ axes give the reduced representation $Z = \tilde X V_k$. I centre always, and I
standardise when the variables have different units. I choose $k$ from the cumulative explained-variance ratio together
with a downstream check.

**Follow-ups.**
- *Why the SVD rather than the eigendecomposition of the covariance matrix?* Forming $\tilde X^\top \tilde X$ squares the condition number. The SVD works on $\tilde X$ directly, and it still works when $n < d$. With two columns that differ by $10^{-7}$, the covariance route was off by $3\%$ to $6\%$ on the small eigenvalue, depending on the BLAS, and the SVD by less than $10^{-10}$ relative (my reproduction).
- *Does the sign of a component matter?* No. Flipping an axis flips its scores too. Only the line matters.

## Q2. When should you standardise before PCA?
**60-second answer.** Whenever the variables are on different scales or units. PCA maximises variance, so a variable
measured in large units dominates the first component for reasons unrelated to structure. Standardising makes PCA
equal to PCA on the correlation matrix, so each variable has unit variance. Don't standardise when the variables share
a unit and raw variance is the signal, as with pixel intensities.

**Follow-ups.**
- *What does standardising cost you?* A near-constant noise variable gets the same weight as an informative one.
  Check for that with a scree plot or parallel analysis.
- *How do you choose $k$?* The cumulative ratio (for example 95%) is a convention. A scree elbow, parallel analysis, or
  the downstream metric are also used. None of them is a theorem.

## Q3. What can go wrong with PCA?
**60-second answer.** Four things. Non-linear structure: a linear subspace can miss a curved manifold. Scale: the axes
depend on units unless you standardise. Outliers: one extreme point can rotate the first axis, because variance is not
robust. Interpretability: each component is a weighted mix of all the inputs. There is also a relevance gap: the top
components maximise variance, not agreement with a label. For prediction, a supervised method can be the better choice.

**Follow-ups.**
- *How would you detect a component driven by outliers?* Refit without the suspect rows, or use a robust covariance
  estimate, and compare the axes.
- *Why is PCA before a linear classifier risky?* It can drop the low-variance direction that carries the class signal.

## Common mistakes
- Skipping the centring step, so the mean can dominate the second moments being decomposed.
- Dividing by $n$ for the variance when the sample variance uses $n-1$, then comparing with a library that uses the other divisor.
- Dividing the explained variance by the kept variance rather than the total, so the ratios sum to 1 for every $k$.
- Treating the sign of a component as meaningful.
- Reading a scree plot without asking whether the variables were on comparable scales.
