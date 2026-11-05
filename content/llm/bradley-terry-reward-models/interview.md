# Interview notes — Bradley-Terry reward models

## Q1. How do you train a reward model from human comparisons?

**60-second answer.** Each training example is a prompt with a chosen and a rejected response. The
model gives each response a scalar reward. Bradley-Terry says the chosen response wins with
probability $\sigma(r_c - r_r)$, and I minimise the mean negative log-likelihood over pairs. I
compute $-\log\sigma(d)$ as $\operatorname{softplus}(-d)$ so large margins do not overflow. I report
pair accuracy with a strict comparison, and I watch the reward scale over training.

**Follow-ups.**
- *Why compare pairs instead of collecting absolute scores?* Raters agree more on which of two
  answers is better than on the absolute value of one answer, and the loss does not need calibrated
  scales.
- *Does the loss fix the reward's offset?* No. Only the difference enters, so any constant added to
  both rewards gives the same loss. The scale is not free: doubling every reward changes each win
  probability, so the data pins it down, unless the pairs are separable and the fit pushes it up without limit. That is
  why I track the reward scale during training.

## Q2. What goes wrong once a policy is optimised against a reward model?

**60-second answer.** The policy searches for outputs that score high on the reward model but that
raters would not prefer. The reward model was fit on comparisons from one distribution, so it is
least reliable where the policy goes. Typical symptoms are longer, more confident or more
formatted text with no gain in quality. Mitigations include a KL penalty to the reference policy,
ensembles, and periodic re-labelling.

**Follow-ups.**
- *What happens on separable preference data?* The loss keeps shrinking as the margins grow, so
  reward magnitudes drift upward. Regularise the reward scale or add a margin term.
- *How do you detect hacking early?* Track the reward on held-out pairs and the reward-to-judge
  gap on samples from the current policy.

## Common mistakes
- Using $\log\sigma(d)$ directly, which returns $-\infty$ for large negative margins.
- Flipping the margin sign, so the loss rewards wrong pairs.
- Averaging the accuracy with ties counted as correct.
- Calling `.detach()` on the rewards, which silently stops the reward model from learning.
