# Interview notes — Epsilon-greedy exploration

## Q1. Why do you need exploration, and what does epsilon-greedy cost?

**60-second answer.** A greedy agent commits to the action with the best estimate. If that estimate is noisy, it can lock onto a suboptimal action forever, since it never samples the others. Epsilon-greedy picks a random action with probability epsilon and the greedy one otherwise. The cost is steady: with a fixed epsilon, every step has an expected loss proportional to epsilon times the gap between the best and average action values, so the regret grows linearly in time.

**Follow-ups.** Why not explore less over time? A decaying schedule makes the exploration cost fall, which is the usual practice once estimates are reliable. How do you choose the schedule? It depends on how fast the estimates settle, and you check it on the regret curve, not on a single run.

## Q2. How would you make a stochastic policy reproducible?

**60-second answer.** Pass one seeded generator, not global state. Fix the number and order of its draws: one uniform decides explore or exploit, and one integer is drawn only when exploring. Break ties among greedy actions by a fixed rule such as the lowest index. Then the same seed and estimates give the same action sequence, and a test can replay the draws exactly.

**Follow-ups.** Why not break ties at random? You can, but then the tie-breaking draws must also be fixed, or the replay changes. Why check frequencies in a test? It catches a wrong probability, such as exploring only among the non-greedy actions, that a single run would not reveal.

## Common mistakes

- Exploring only among non-greedy actions. Epsilon-greedy explores uniformly over all actions, so the greedy one can also come up.
- Drawing a random number on every call and so making the draw order depend on the branch taken.
- Using the global random state, so that unrelated code changes the actions.
