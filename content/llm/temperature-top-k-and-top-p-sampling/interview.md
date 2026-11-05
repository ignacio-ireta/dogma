# Interview notes: Temperature, top-k and top-p sampling

## Q1. How do temperature, top-k and top-p interact when you pick the next token?

**60-second answer.** Temperature divides the logits before the softmax, which sets how peaked the distribution is. Top-k keeps a fixed number of candidates, and top-p keeps the smallest set whose probability mass reaches a threshold. Apply them in that order, renormalise the survivors, and then draw. Top-p is the more adaptive filter because the candidate count depends on the model's confidence, whereas top-k always keeps the same count.

**Follow-ups.**

- Why not use temperature 0? Dividing by zero is undefined. Greedy decoding is the separate argmax rule, and it is what temperature 0 converges to.
- What is the failure mode of a flat distribution with top-k? The fixed count keeps many weak candidates, which is noisy. Top-p keeps as many as needed for the mass, which can be many too, so it does not remove the noise by itself.

## Q2. How would you make sampled generation reproducible and testable?

**60-second answer.** Pass an explicit `torch.Generator` seeded per request, and never draw from the global random state. Test the filter separately from the draw: the filtered distribution is deterministic and checked against a reference, and the draw is checked by its empirical frequencies against a tolerance justified by the sample size.

**Follow-ups.**

- How big a sample do you need for a frequency test? Choose the tolerance from the binomial standard error, sqrt(p(1-p)/n), and use a few standard errors. A seed makes the check deterministic.
- Where does the randomness go in a production stack? Each request carries its own generator, so a batch of requests does not share one stream.

## Common mistakes

- Dividing by temperature after the softmax instead of before.
- Using $>$ where the top-p rule needs $\ge$.
- Keeping every tied token in top-k, so more than $k$ survive.
- Forgetting to renormalise after a filter.
- Drawing with the global random state, so runs cannot be repeated.
