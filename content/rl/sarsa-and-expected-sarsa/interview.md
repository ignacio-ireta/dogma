# Interview notes: Sarsa and expected Sarsa

## Q1. How does Sarsa differ from Q-learning?

**60-second answer.** Both update one table entry toward a one-step target, and they differ only
in the next-state term. Sarsa uses the value of the action the agent actually takes next, $Q(s',a')$,
so it learns the value of the policy it follows, exploration included. Q-learning uses
$\max_b Q(s',b)$, so it learns the greedy policy's values while the agent behaves differently. The
difference matters when the behaviour is exploratory: Sarsa's values include the cost of the
random actions it will take.

**Follow-ups.**
- *When are they the same?* When the next action is greedy. At $\varepsilon = 0$ that holds in every
  step, up to ties.
- *Which one converges to the optimum?* Q-learning does under standard step-size conditions and
  infinite visits. Sarsa does when the policy becomes greedy in the limit while still exploring.

## Q2. What does expected Sarsa change, and what does it cost?

**60-second answer.** It replaces the sampled next action with its expectation under the policy:
$r + \gamma \sum_b \pi(b\mid s')\,Q(s',b)$. Given the transition, that removes the variance of the
next-action draw, and the target's variance can only shrink. The cost is one sum over the actions per
update, instead of one table lookup. At $\varepsilon = 0$ it is the Q-learning target, and its
target is never above Q-learning's for the same table.

**Follow-ups.**
- *Why is the variance lower?* The law of total variance splits Sarsa's target variance into the
  expected target's variance plus the average variance over the next action.
- *Does the tie-breaking rule matter?* Not for this target. Tied actions have equal values, so only
  the total mass on the greedy set matters. It does change which action the sampler favours.

## Q3. When would you choose Sarsa over Q-learning?

**60-second answer.** When the learning-time behaviour has a cost. Sarsa's target includes the
exploratory actions, which is why it is often described as the safer learner near risky states.
Q-learning is the choice when the goal is the best greedy policy regardless of how the data were
gathered. Both need a suitable step size schedule and enough exploration. The safety claim is a
standard illustration that this course does not check, and the repository has no primary text for it.

**Follow-ups.**
- *Off-policy or on-policy?* Q-learning is off-policy, because its target evaluates a different
  policy from the one that generates the data. Sarsa and expected Sarsa are on-policy.
- *Can Sarsa use the action from a different policy?* Only if the target is weighted by importance
  ratios, which is a separate method.

## Common mistakes

- Resampling $a'$ for the update, so the target describes an action the agent does not take.
- Writing the max into the Sarsa target. That is Q-learning under another name.
- Indexing `Q[None, a']` for a terminal next state, instead of using the reward alone.
- Dropping the $\varepsilon/|\mathcal A|$ share on greedy actions, so the policy weights sum to less than 1.
- Treating near-equal values as ties with a tolerance, which changes the policy.
