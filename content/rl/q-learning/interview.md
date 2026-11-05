# Interview notes — Q-learning

## Q1. Write the Q-learning update and explain why it is off-policy.

**60-second answer.** After taking action a in state s, receiving r, and landing in s', update Q(s,a) toward r + gamma times the max over a' of Q(s',a'). At a terminal s' the max term is dropped. The target uses the greedy value of the next state, so it estimates the optimal action values regardless of which action the agent actually takes next. The behaviour policy can be epsilon-greedy for exploration while the target policy is the greedy one. That gap is what makes the method off-policy.

**Follow-ups.** How does SARSA differ? SARSA bootstraps from the action the behaviour policy takes, so it learns the value of the exploring policy. Why drop the bootstrap at terminal states? A terminal state has no future rewards, so adding a value there would count returns that do not exist.

## Q2. When does tabular Q-learning converge, and what does that rely on?

**60-second answer.** Under Robbins-Monro step sizes, which are positive, sum to infinity, and have a finite sum of squares, with every state-action pair visited infinitely often, the tabular estimates converge to the optimal action values. These are conditions on a limit, not a guarantee for a fixed finite run. A constant step size gives tracking and no such convergence in the strict sense. Function approximation can diverge; that is the deadly-triad concern in `rl/the-deadly-triad`.

**Follow-ups.** Why does exploration matter for convergence? A pair that is never visited never receives an update. How would you check a finite run? Compare the values against a known optimum for a small problem, or measure the Bellman residual over visited pairs.

## Common mistakes

- Bootstrapping from the next state after the episode ends, which leaks a value that does not exist.
- Using the action the behaviour policy takes next, which turns the update into SARSA.
- Declaring convergence from a single run without checking visitation counts.
