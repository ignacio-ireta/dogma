# Interview notes — Direct preference optimization (DPO)

## Q1. How does DPO replace RLHF, and what does the loss look like?

**60-second answer.** RLHF trains a reward model and then runs RL against it with a KL penalty to the
reference. DPO shows that the KL-regularised optimum implies a reward that is $\beta$ times the
log-ratio of policy to reference, plus a term that depends only on the prompt. Inside a pair that
prompt term cancels, so the loss is the Bradley-Terry loss on the implicit rewards:
$-\log\sigma(\hat r_c - \hat r_r)$. There is no reward model and no sampling during training. The
data is pairs of responses, and the gradient comes from log-probabilities of those responses.

**Follow-ups.**
- *What does $\beta$ control?* The strength of the anchor to the reference. A large $\beta$ keeps the
  policy near the reference, and it also saturates the sigmoid on smaller log-ratio gaps. A small
  $\beta$ lets the policy drift.
- *Why sums and not means of token log-probs?* The implicit reward is defined on the whole response.
  A sum is the right object for that. It also makes the margin depend on length, which is why
  length-normalised variants exist.

## Q2. What must be true about the reference model and the masks?

**60-second answer.** The reference must score exactly the same tokens as the policy, under the same
template and the same response mask, and it must be frozen. If the reference scores a different
span, its log-probs add a response-specific offset to each implicit reward. That offset does not
cancel between chosen and rejected, so the loss trains on an offset that is not a preference. Gradients must never flow into the reference.

**Follow-ups.**
- *What is the one-token shift for?* The logits at position $t$ score token $t+1$. A mask that is not
  shifted scores the wrong tokens.
- *What failure should you watch for?* Both chosen and rejected log-probs can fall while the margin
  grows. Track the chosen log-probability separately from the margin.

## Common mistakes
- Using the policy log-probs without subtracting the reference, which gives a loss with no anchor.
- Averaging token log-probs when the definition is a sum over response tokens.
- Dropping $\beta$ from one of the two implicit rewards.
- Including prompt tokens in the sum.
