## Hint 1 — concept

Layer normalization is batch normalization with the reduction moved onto the features of one example. The affine gain and bias are still per feature, and they broadcast over every leading position.

## Hint 2 — formula / approach

For each vector along the last dimension, compute the mean $\mu$, subtract it, then divide the centred values by $\sqrt{\sigma^2+\epsilon}$, where $\sigma^2$ is the biased variance over $D$. Multiply by the gain and add the bias. Subtracting the mean before squaring is what keeps the variance accurate when the inputs share a large offset.

## Hint 3 — code shape

Validate first: floating dtype, at least one dimension, a nonzero last dimension, weight and bias of shape `(D,)`, and a positive eps. Then use `mean(dim=-1, keepdim=True)` and `keepdim` on the variance so the broadcasting is explicit. Return a new tensor and leave `x` untouched.
