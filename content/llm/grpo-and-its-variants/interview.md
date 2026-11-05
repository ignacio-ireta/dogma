# Interview notes — GRPO and its variants

## Q1. How does GRPO get a baseline without a value model?

**60-second answer.** For each prompt, GRPO samples a group of responses and scores them. The mean of
the group rewards is the baseline, and the advantage of each response is its reward minus that mean,
divided by the group standard deviation. PPO needs a critic to estimate a value for every partial
response. For long reasoning outputs with one final reward, that critic is hard to fit and costs
about as much as the policy. GRPO drops it and pays with $G$ samples per prompt.

**Follow-ups.**
- *What do you lose by dropping the critic?* Per-token credit: every token in a response gets the same
  advantage. A critic can in principle assign different credit along the sequence.
- *Why divide by the standard deviation?* It puts prompts with different reward spreads on a common
  scale. It also amplifies prompts whose spread is small, which can be noisy.

## Q2. What do the clip and the KL term each do?

**60-second answer.** The clip bounds the probability ratio of a token, so one update cannot move a
token far from the policy that generated the sample. With a positive advantage, a ratio above $1+\varepsilon$
gives no further gradient. The KL term keeps the policy near the reference. GRPO uses the $k_3$
estimator, which is non-negative for every sample and unbiased for the KL when samples come from the
policy.

**Follow-ups.**
- *What is the all-equal group case?* Zero advantage, so only the KL term acts on that prompt.
- *Where can the objective go wrong?* Reward hacking, or a group size too small for a stable mean.

## Common mistakes
- Using the sample standard deviation without saying so.
- Dividing by zero for an all-equal group, which gives NaN instead of zero.
- Averaging over the whole batch of tokens when the convention is per response.
- Using the naive KL $\log(\pi_\theta/\pi_{\text{ref}})$, which is not guaranteed non-negative per sample.
