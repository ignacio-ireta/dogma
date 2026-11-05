## Hint 1 — concept

The elastic net update is the lasso update with two changes. The threshold uses only the L1 share of
α. The L2 share of α is added to the curvature in the denominator. That extra curvature is what makes
correlated columns share their weight, where the lasso picks one of them.

## Hint 2 — formula / approach

Let the threshold be α·ρ, where ρ is `l1_ratio`, and the L2 term be α·(1 − ρ). For coordinate j the new
coefficient is the soft-threshold of ρⱼ at the threshold, divided by zⱼ plus the L2 term. Skip a
coordinate whose denominator is zero, which happens for a zero column when there is no L2 term.

## Hint 3 — code shape

Compute `threshold` and `l2_term` once, before the sweep loop. Inside the loop, form
`denominator = z[j] + l2_term`, guard it against zero, and otherwise reuse the lasso loop structure from
the previous problem with the new threshold and denominator.
