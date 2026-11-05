# Interview notes — Softmax

## Q1. Implement a numerically stable softmax. What exactly do you subtract, and why is the answer unchanged?

**60-second answer.** Subtract the maximum of each slice along the normalised axis, keeping that axis so the maximum broadcasts back onto its own slice. Softmax is invariant to adding a constant to every logit in a slice, because the constant cancels between numerator and denominator. After the shift every exponent is at most zero, so none overflows, and the largest term is exactly one, so the normaliser cannot underflow to zero.

**Follow-ups.**
- *Why not subtract the maximum over the whole matrix?* Rows with much smaller logits are shifted into underflow. Their normaliser becomes zero, which gives NaN.
- *What if the matrix is square?* Forgetting `keepdims` still broadcasts, but along the wrong dimension. The result is silently wrong, so test a square case.

## Q2. Why do classifiers use log-softmax in the loss rather than log of softmax?

**60-second answer.** The log of a probability that has been rounded can be $\log 0 = -\infty$. Log-softmax computes the logit minus the log-sum-exp, keeping the calculation in shifted coordinates. Representable log-probabilities stay finite; a log-probability below float64 range is necessarily $-\infty$. A confident wrong prediction then has a large but finite loss, and its gradient stays correct. This is the same reason the cross-entropy is computed from logits.

**Follow-ups.**
- *What does temperature do?* It rescales the logits. At small $T$ the distribution concentrates on the maximum, and at large $T$ it flattens toward uniform.
- *How does this relate to the sigmoid?* Softmax of $(z, 0)$ is $(\sigma(z), 1 - \sigma(z))$.

## Common mistakes

- Using a global maximum, or no maximum, when the batch has rows of very different scales.
- Dropping `keepdims`, which gives a silently wrong answer for square inputs.
- Adding a probability floor to avoid $\log 0$, which breaks normalisation.
- Taking `np.log(softmax(z))` in a loss.
