## Hint 1 — concept

The update changes one number, the value of the action just taken, toward a target built from
the reward and the best value available in the next state. The terminal flag switches the
second part off. Almost every failure in this problem is one of those three ideas done wrong:
the wrong entry, a leaked next-state value, or a draw made in the wrong order.

## Hint 2 — formula / approach

For each transition, the target is the reward if the next state is terminal, and the reward plus
gamma times the largest next-state value otherwise. The new value is the old value plus alpha
times the difference between target and old value. For the run, use the epsilon-greedy draw
order in the problem: one uniform on every choice, then one integer only when exploring, then the
lowest-index greedy action. Reset to the start state at each episode, and stop at the goal.

## Hint 3 — code shape

Write the update as a function that copies the table, computes the target with a conditional on
the terminal flag, and applies the change to the one copied entry. The run is two nested loops: an
outer loop over episodes that resets the state, and an inner loop of at most the step cap. Inside
the inner loop, choose the action in an if/else on the uniform draw, step the grid with a small
helper that returns the next state, the reward and the terminal flag, then call the update and
advance the state.
