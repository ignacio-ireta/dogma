# Interview notes — REINFORCE

## Q1. Derive the policy gradient estimator and explain the loss you would code.

**60-second answer.** The objective is the expected return under a parametric policy. The score-function identity turns its gradient into an expectation of the return times the gradient of the log-probability of the sampled action. A sampled episode gives an unbiased estimate. In code, the loss is the negative sum of log-probabilities weighted by reward-to-go, averaged over episodes. Minimising it ascends the expected return. The estimator has high variance, which is the reason for baselines.

**Follow-ups.** Why reward-to-go? Actions cannot affect earlier rewards, so the earlier rewards only add variance. What is the role of the minus sign? Autograd minimises, and the objective is to maximise.

## Q2. How do baselines and return normalisation change the estimator?

**60-second answer.** A baseline that does not depend on the action being weighted leaves the expected gradient unchanged, because the expectation of the score function times such a term is zero. It reduces variance when the baseline tracks the typical return. A batch mean over the same batch is common, but it depends on the batch, so the strict guarantee needs a baseline fitted without the episode itself. Normalising returns to zero mean and unit spread is a practical scale fix. It stabilises the step size, but it changes the weighting and is not a pure variance trick.

**Follow-ups.** Why does a learned value baseline help? It approximates the expected return from each state, so the advantage is closer to zero and the variance drops. What can go wrong? A poor baseline adds variance, and a leaky one can add bias.

## Common mistakes

- Using the total episode return for every step, which ignores causality and adds variance.
- Taking the log of a softmax in two steps, which overflows or yields negative infinity for extreme logits.
- Dropping the sign, so the loss descends the return.
