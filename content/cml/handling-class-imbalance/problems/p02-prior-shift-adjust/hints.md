## Hint 1 — concept

Under a prior shift, the evidence from the features does not change, only the class mix does. Posterior odds are prior odds times the likelihood ratio. Changing the prior therefore multiplies the odds by one fixed number, which you can compute from the two priors alone.

## Hint 2 — formula / approach

Convert both priors to odds, `prior / (1 - prior)`, and divide the deployment odds by the training odds to get the ratio r. Multiply each probability's odds by r and convert back. Written as `r p / (r p + 1 - p)`, the conversion needs no division by `1 - p`, so the endpoints are safe.

## Hint 3 — code shape

Convert `probs` to a float array and validate shape, finiteness and bounds. Validate both priors. For extreme priors, form the difference of their log odds using `log` and `log1p`, then add it to the interior probabilities' log odds. Convert back with a sigmoid whose exponential argument is non-positive. Handle exact zero and one separately, and return a new float array.
