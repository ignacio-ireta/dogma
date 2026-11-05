## Hint 1 — concept

Run the graph forward once and store each intermediate value. Then walk the steps backwards. At
each step, multiply the step's local derivative by the gradient that reached it from the output.
When a value feeds two later steps, the gradient it receives is the sum of the two contributions.

## Hint 2 — formula / approach

Start with the adjoint of the loss, which is 1. For `L = r * r`, the adjoint that reaches `r` is
`2 * r`, because both factors are `r`. Each later step then passes its adjoint to its inputs:
the local derivatives of `y = q + e`, `r = y - t`, `z = p + b` and `h = tanh(z)` are 1, -1, 1 and
`1 - h*h`. `h` and `w` each have two consumers, so each collects two terms.

## Hint 3 — code shape

Compute the forward values in order (`p`, `z`, `h`, `q`, `e`, `y`, `r`, `loss`). Then write the
backward pass in reverse order with one variable per adjoint: `g_r`, `g_y`, `g_t`, `g_q`, `g_e`,
then `g_h`, which is a sum of two terms, then `g_z`, `g_b`, `g_p`, `g_x`, and finally `g_w`,
which is a sum of two terms. Check each sum by testing whether the path it adds is really there.
