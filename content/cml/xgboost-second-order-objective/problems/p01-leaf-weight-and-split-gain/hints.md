## Hint 1 — concept

Each leaf's weight is the value that minimises a quadratic approximation of the loss over the samples in that leaf, and the regulariser enters that quadratic as an extra curvature term. A split is good when the two children score better than the parent. Every leaf costs a fixed amount, `gamma`, so a split must beat that cost to be worth keeping.

## Hint 2 — formula / approach

Sum the gradients and the hessians on each side to get $G_L, H_L, G_R, H_R$. The leaf weight of a side is $-G/(H+\lambda)$. The gain is built from the same quantity, $G^2/(H+\lambda)$, evaluated for the left side, the right side and the parent. Subtract the parent term from the two children's terms, halve the result, then subtract $\gamma$.

## Hint 3 — code shape

Validate first: each regulariser is finite and non-negative, each pair of arrays is 1-D, equal in length, finite, and the hessians are non-negative. Check that each side has $H+\lambda > 0$. Then compute the four sums, the two leaf weights, the parent term from the summed sides, and the gain. Return three Python floats.
