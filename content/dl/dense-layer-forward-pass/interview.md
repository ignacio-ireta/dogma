# Interview notes — Dense layer forward pass

## Q1. Walk me through a dense layer on a batch.

**60-second answer.** I fix the storage convention first: X is (n, input), W is (input, output), and b is (output,). Each output entry is one dot product plus its output-specific bias. The batch shares W and b, so processing more examples changes arithmetic and activation storage, not the parameter count. I validate shapes before computing because broadcasting can hide an incorrect interface.

**Follow-ups.**

- What about a single example? Keep a leading batch dimension of one.
- Does zero input width make sense? Yes in this exercise: the empty sum leaves a repeated bias.

## Q2. How do you diagnose a silent shape bug?

**60-second answer.** Write what each axis means, then calculate one output entry by hand. A square weight can hide an unintended transpose, and a bias can broadcast along the wrong axis when dimensions coincide. I use a nonsquare example and a batch size that exercises broadcasting, then check that changing one row affects only that row.

**Follow-ups.**

- What if another framework stores weights differently? Transpose at the interface after checking its documented layout.
- Can two affine layers replace a nonlinear network? Without an activation they collapse to one affine map.

## Common mistakes

- Treating multiplication success as proof of correct orientation.
- Accepting a two-dimensional bias when the interface requires a vector.
- Mutating an input view while adding the bias.
