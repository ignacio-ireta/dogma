## Hint 1 — concept

Every pair of a query point and a reference point is a difference of coordinates. You can build all differences at once by giving the two inputs axes of size 1 where the other has a size. Then no loop over pairs is needed.

## Hint 2 — formula / approach

Let $X$ have shape $(B,N,D)$ and $Y$ have shape $(B,M,D)$. Inserting a size-1 axis gives $X$ the shape $(B,N,1,D)$ and $Y$ the shape $(B,1,M,D)$. Their difference broadcasts to $(B,N,M,D)$. Square it and sum over the last axis. The expansion $\|x\|^2-2x\cdot y+\|y\|^2$ is cheaper but can go negative in floating point, so compare its cancellation with the difference form before you choose it.

## Hint 3 — code shape

Promote both inputs to the same dtype with `torch.promote_types` before any arithmetic. Validate the rank, the float dtype and the matching batch and feature sizes first. Compute the difference with `None` indexing, square it, and sum over the last axis.
