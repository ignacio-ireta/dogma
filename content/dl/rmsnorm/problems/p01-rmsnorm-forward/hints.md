## Hint 1 — concept

RMSNorm is layer normalization with one statistic removed. The scale is set by the average squared size of the features, not by their spread around the mean. Nothing is shifted.

## Hint 2 — formula / approach

For each vector along the last dimension, compute the mean of the squares $\overline{x^2}$. Divide $x$ by $\sqrt{\overline{x^2}+\epsilon}$, with eps inside the root, and multiply by the gain. Do not subtract a mean anywhere.

## Hint 3 — code shape

Validate first, as in layer norm: a floating dtype, at least one dimension, a nonzero last dimension, gain of shape `(D,)` and a positive eps. Then `mean(dim=-1, keepdim=True)` of the squares, one division by the root, and one multiplication by the gain. Return a new tensor.
