## Hint 1 — concept

The loss is a weighted sum of log-probabilities of the actions taken. Each weight is a return
from that step onward, optionally centred across the batch. Two things need care: the returns
must be computed backwards from the last step, and the log-probabilities must come from a
numerically stable log-softmax. Gradients flow only through the log-probabilities.

## Hint 2 — formula / approach

Compute the reward-to-go with the backward recurrence $G_t = r_t + \gamma G_{t+1}$, where $G$ is
zero after the last step. If a baseline is requested, subtract the mean over episodes at each step
from $G$. Take a log-softmax of the logits over the action axis, pick out the log-probability of
each taken action, multiply by the advantage, sum, negate and divide by the number of episodes.

## Hint 3 — code shape

Validate the inputs first, and convert the discount to a float. Build the returns in a loop over the
step index from last to first, keeping one running value per episode. Then branch on the baseline
flag to form the advantage. Get the log-probabilities with `log_softmax` on the last dimension, then
select the taken action with a gather along that dimension and remove the singleton. Finish with a
single expression for the weighted sum.
