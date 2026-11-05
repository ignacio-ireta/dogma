# Interview notes — Action-value estimates

## Q1. How would you estimate the value of an action from samples, and why does the update look like that?

**60-second answer.** Keep one estimate per action. For the sample average, store the count and move the estimate by one over the count times the error: Q <- Q + (1/n)(r - Q). That is algebraically the running mean, so it needs no history and costs constant memory per action. It is the general form NewEstimate <- OldEstimate + StepSize*(Target - OldEstimate), with the step size chosen as 1/n.

**Follow-ups.** Why not store all rewards? The incremental form gives the same number with O(1) memory and time per sample. When would you use a constant step size instead? When the reward distribution drifts, because a mean over all history reacts too slowly to change.

## Q2. What is the effect of a constant step size, and what is the risk of optimistic initial values?

**60-second answer.** A constant alpha gives weight alpha*(1 - alpha)^k to the reward k updates ago. Old rewards decay geometrically, so the estimate tracks change, but it never settles to a fixed value. An optimistic initial value above the true reward makes untried actions look attractive. That buys exploration early. It is a bias that vanishes only if each action is tried enough times to overwrite the initial value.

**Follow-ups.** Can the sample average forget the initial value? Only after its first update, since alpha for the first update is one. Is an optimistic start a substitute for exploration? No. It helps only while untried actions remain, and it fails when the optimistic value is below the true reward.

## Common mistakes

- Updating every action's estimate on each step instead of only the action taken.
- Using 1/(n+1) or 1/(n-1), which is an off-by-one in the count.
- Forgetting that the first sample-average step is one, so the initial value is forgotten immediately.
