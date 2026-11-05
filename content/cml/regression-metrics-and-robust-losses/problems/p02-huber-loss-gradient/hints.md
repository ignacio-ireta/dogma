## Hint 1 — concept

The Huber loss treats small residuals as squared error and large ones as absolute error. Its
derivative is therefore capped: once a residual is past the threshold, it stops pulling harder. The
gradient you return is that derivative, scaled by 1/n because the loss is a mean.

## Hint 2 — formula / approach

Split every residual on the test |r| ≤ δ. Inside the threshold the loss is ½r² and its derivative is r.
Outside it the loss is δ(|r| − δ/2) and its derivative is δ·sign(r). The two pieces agree at |r| = δ.
Average the losses over n, and divide each derivative by n. Validate the inputs before computing.

## Hint 3 — code shape

Allocate the per-residual losses, then use boolean masks to evaluate each branch only where it applies. `np.where` would also square huge linear-branch residuals. Average safely: a sum can overflow even when its mean is representable. Divide the derivative by `n`, and validate `delta` before computing.
