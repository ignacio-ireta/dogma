## Hint 1 — concept

The value of a state under a fixed policy is the expected reward now, plus the discounted value
of wherever you land next. Two averages are hidden in that sentence: one over the actions the
policy picks, and one over the next states the transitions pick. Averaging the actions first
turns the problem into a smaller one with a single transition matrix.

## Hint 2 — formula / approach

Build the policy-averaged transition matrix $P_\pi[s,s']=\sum_a \pi(a\mid s)P(s'\mid s,a)$ and
the policy-averaged reward vector $R_\pi[s]=\sum_a \pi(a\mid s)R(s,a)$. Then repeat
$V \leftarrow R_\pi + \gamma P_\pi V$ from zero. Each sweep is a contraction by $\gamma$ in the
max norm, so the change between sweeps shrinks geometrically. Stop when the largest change is
below the tolerance.

## Hint 3 — code shape

Use one contraction over the action axis for each averaged array, with the policy as one of the
operands. Put the validation in helper functions that raise `ValueError` before any iteration.
Inside the loop, compute the next vector from the previous one, measure the largest absolute
difference, then replace the old vector and test the threshold.
