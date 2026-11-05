## Hint 1 — concept

The return from step t is not a new sum for each t. It is the next return, discounted once,
plus the reward that follows the action at step t. Ask which values you already know when
you start at the last step.

## Hint 2 — formula / approach

Use the backward recurrence $G_t = r_{t+1} + \gamma\,G_{t+1}$, where $G_T = 0$. In the code,
$r_{t+1}$ is `rewards[t]`, the reward after the action at step $t$. Walk from the
last index to the first, keeping one running value. Each step overwrites that value and stores
it in the output at the same index. Validate the input before the loop: the dtype, the
dimension and the finite values, then gamma.

## Hint 3 — code shape

Allocate a float64 output with one slot per step. Start a running value at zero. Loop over the
step indices from last to first. In each pass, combine the reward at that index with the
discounted running value, store the result in the output, and keep it as the new running value.
Return the output.
