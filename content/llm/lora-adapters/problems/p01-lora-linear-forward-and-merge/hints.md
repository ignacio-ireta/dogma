## Hint 1 — concept

The base layer and the adapter are two paths that add into one output. The base path is the frozen
weight you were given. The adapter path is a small product through a bottleneck of width `r`. Autograd
must see both paths, so every operation on `A` and `B` has to stay in the graph.

## Hint 2 — formula / approach

Let `r = A.shape[0]` and `scale = alpha / r`. The unmerged output is
`x @ W.T + scale * ((x @ A.T) @ B.T)`, plus the bias once. The merged output first forms
`W_eff = W + scale * (B @ A)`, which has the shape of `W`, and then applies `x @ W_eff.T`. Both must
give the same numbers. Check every rule in the contract before any arithmetic: the ranks, the shapes,
the bias, `alpha`, and the dtypes.

## Hint 3 — code shape

Validate in this order: `x` has a last dimension, `W` is 2-D, `A` has width `d_in`, `B` has shape
`(d_out, r)`, the bias is `None` or `(d_out,)`, `alpha` is finite, and all floating tensors share one
dtype. Then branch on `merge`. Each branch returns a `(..., d_out)` tensor, and the bias is added after
the branch. Do not write into `W` in place.
