# Interview notes — Residual connections

## Q1. Why do residual connections make very deep networks trainable?

**60-second answer.** A residual block computes y = x + F(x), so it learns a correction on top of an identity mapping. A deeper stack can copy a shallower solution by driving the new branches toward zero, which avoids the degradation He et al. observed in plain networks. For the gradient, the Jacobian is dy/dx = I + dF/dx, so every block passes a unit-gain path backward. The residual form does not make optimisation trivial, but it removes the easy failure of multiplying many small Jacobians.

**Follow-ups.** Is this overfitting? No: the plain deeper net had higher training error. Does the identity path guarantee a bounded gradient? No; branch terms can still grow or cancel.

## Q2. A transformer uses pre-norm. Why, and what does it cost?

**60-second answer.** Pre-norm writes x + F(N(x)), so the identity path is never normalised and the gradient reaches early layers unchanged. Post-norm normalises after the addition, and the original Transformer used a learning-rate warm-up for it. Pre-norm is reported to be easier to train without that warm-up. The reported cost is sometimes a slightly weaker final performance than a well-tuned post-norm model; check the source before quoting that.

**Follow-ups.** Where does the normalisation go in the final layer? Pre-norm stacks often add a final normalisation after the last block. Does it change the parameter count? No; it changes only the ordering of operations.

## Q3. When does a residual shortcut need a projection?

**60-second answer.** When the branch changes the width, the identity is undefined, so the shortcut must be a learned linear map of shape (d_in, d_out), as in ResNet's option B. Projections are not required to fix degradation; He et al. report that identity shortcuts plus zero-padding work well for the same-width blocks.

**Follow-ups.** Why not zero-pad? Padding silently drops or invents features. Why not always project? Extra parameters and compute with little benefit when widths already match.

## Common mistakes

- Applying the activation after the addition, which breaks the clean identity path.
- Detaching the shortcut, so the forward output is correct but the input gradient misses I.
- Silently cropping or padding when widths differ instead of projecting.
