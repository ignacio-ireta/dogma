## Hint 1 — concept

Each pair is a two-way choice. The model turns the difference of the two rewards into a probability,
and the loss is the negative log of the probability the rater's choice received. Only the difference
between the two rewards is used.

## Hint 2 — formula / approach

With margin $d = r_c - r_r$, the per-pair loss is $-\log \sigma(d) = \operatorname{softplus}(-d)$.
Use the split form $\max(x, 0) + \log(1 + e^{-|x|})$ with $x = -d$ so the exponential never sees a
large positive argument. Write the positive part as $(x + |x|) / 2$, not with `clamp`: both have the
same values, but `clamp` has a kink at 0 and gives a tied pair no gradient, where the loss has slope
1/2. The accuracy is the mean of the indicator $d > 0$, cast to the input dtype.

## Hint 3 — code shape

Validate first: both arguments are tensors, 1-D, the same shape, non-empty, floating-point, the same
dtype, and finite. Then compute the margin, the per-pair loss in the split form, its mean, and the
strict comparison for accuracy. Return two 0-dimensional tensors in the input dtype. Do not call
`.detach()` or `.item()`, because the loss must stay differentiable.
