# Interview notes — Backprop through a chain of scalars

## Q1. Explain backpropagation through a reused value.

**60-second answer.** I evaluate the graph forward and store what the local derivatives need. Starting with loss adjoint one, each edge passes its received gradient times its local derivative to its input. When an input has several uses, I add all those contributions. In this graph w appears in w*x and h*w, so the final w gradient includes both a direct path and an indirect path through h.

**Follow-ups.**

- Why add at a branch? The loss changes through every downstream use, and the chain rule sums those paths.
- Is an optimizer part of backprop? No; backward computes sensitivities, and an optimizer later uses them.

## Q2. How do you establish that the gradients are correct?

**60-second answer.** I compare with a closed-form expression and with central differences of an independent forward loss. I vary one scalar input while holding the others fixed. Reuse cases catch overwritten contributions; a zero residual checks that all loss gradients vanish; saturation checks that the direct weight path survives when tanh derivative vanishes. Finite differences need a sensible step and tolerance because floating point error and curvature compete.

**Follow-ups.**

- Why not make the difference step arbitrarily small? Subtraction and rounding error eventually dominate.
- Why reverse mode for network training? A scalar loss allows intermediate derivatives to serve many input-parameter gradients in one reverse traversal.

## Common mistakes

- Overwriting an existing adjoint instead of accumulating.
- Forgetting that r*r has two contributions from r.
- Erasing a direct weight path when tanh saturates.
