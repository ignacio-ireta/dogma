# Interview notes — Cross-entropy

## Q1. Write the cross-entropy loss and its gradient for a batch of logits. How do you compute it?

**60-second answer.** For each row take the negative log of the softmax probability of the true class, then average over rows. Compute the log from the logits with a log-sum-exp: the loss for a row is $\operatorname{logsumexp}(\mathbf{z}) - z_t$. The gradient with respect to the logits is $(\mathbf{p} - \mathbf{e}_t)/n$, the predicted probabilities minus the one-hot target, divided by the batch size. Each row sums to zero, so the gradient moves probability mass from the other classes to the true one.

**Follow-ups.**
- *Why not `log(softmax(z))`?* A probability of a wrong class can round to 0, and the log is then $-\infty$.
- *What do you check in a test?* A finite-difference gradient on a small problem, and a sum-versus-mean check by repeating a row.

## Q2. Why does a sigmoid output trained with binary cross-entropy not suffer from vanishing gradients?

**60-second answer.** With squared error the gradient carries the factor $\sigma'(z) = \sigma(z)(1-\sigma(z))$, which is near zero when the unit saturates, however wrong it is. Binary cross-entropy has the same chain rule, but its derivative with respect to the logit is $\sigma(z) - y$, and the $\sigma'$ cancels. A confidently wrong unit therefore gets a gradient close to $\pm 1$.

**Follow-ups.**
- *How do you compute the loss stably?* Use $\max(z, 0) - zy + \log(1 + e^{-|z|})$, which never forms a probability.
- *What does label smoothing change?* It replaces the target with positive probabilities, which favours finite logit differences and limits overconfidence; a common logit offset is still free.

## Common mistakes

- Taking the log of a softmax probability, which underflows for confident predictions.
- Summing over the batch when the gradient is scaled by a mean, so the loss and the gradient disagree.
- Casting float class labels to integers, which truncates them silently.
- Forgetting that the clip of a probability changes the loss for confident examples.
