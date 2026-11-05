## Hint 1 — concept

Each action has its own running estimate and its own count. A pull changes only the estimate
of the action that was pulled. The count answers one question: how many rewards, including this
one, has this action produced so far?

## Hint 2 — formula / approach

For the sample average, the step size on the $n$th pull is $1/n$, with $n$ counting that pull.
So the first pull of an action has step size one. For a constant step size, use $\alpha$ on
every pull. In both cases the update is $Q \leftarrow Q + \text{step}\,(r - Q)$, where $Q$ is the
estimate for the pulled action before this pull.

## Hint 3 — code shape

Validate the arguments first: the number of actions, the initial value, the step size, then the
arrays. Allocate one float estimate per action, all set to the initial value, and one integer count
per action set to zero. Walk through the pulls in order. For each pull, increment that action's
count, choose the step from the count or from the constant, then move the pulled estimate toward
the reward. Return the estimates.
