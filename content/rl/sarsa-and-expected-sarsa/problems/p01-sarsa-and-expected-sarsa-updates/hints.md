## Hint 1: concept

Both updates move one table entry a fraction `alpha` of the way toward a target. They differ
only in the target's next-state term. Sarsa uses the one action the agent takes next. Expected
Sarsa uses the average of the next state's values under the epsilon-greedy policy. A terminal
next state has no next-state term at all.

## Hint 2: formula / approach

The target is `r + gamma * value` where `value` is `Q[s_next, a_next]` for Sarsa and
`sum_b pi(b) * Q[s_next, b]` for expected Sarsa. Each epsilon-greedy weight is `epsilon / A`,
plus `(1 - epsilon) / k` if the action is among the `k` largest values of `Q[s_next]`. Check that
the weights sum to 1. Validate all inputs before any arithmetic, and copy the table before
writing into it.

## Hint 3: code shape

Compute `greedy = row == row.max()`, then `weights = np.full(A, epsilon / A)` and
`weights[greedy] += (1 - epsilon) / greedy.sum()`. Take `weights @ row` for the expectation.
Return a copy of `Q` with entry `(s, a)` replaced. Write the terminal branch first, so the
`a_next` argument is never read when `s_next is None`.
