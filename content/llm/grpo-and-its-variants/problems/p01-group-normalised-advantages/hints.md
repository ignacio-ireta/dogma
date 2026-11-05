## Hint 1 — concept

Each response is judged against the other responses sampled for the same prompt. The group supplies
the baseline and the scale, so no separate model is needed. Work one prompt at a time, and keep the
groups independent of each other.

## Hint 2 — formula / approach

For a row of $G$ rewards, the advantage is the reward minus the row mean, divided by the row's
population standard deviation plus $\epsilon$. The population variance is the mean of squared
deviations, dividing by $G$. A row where all rewards are equal has zero deviations, so the advantages
are zero and the division by $\epsilon$ keeps the result finite.

## Hint 3 — code shape

Validate first: a 2-D floating tensor, at least one row, at least two columns, finite values, and a
positive finite `eps`. Then take the mean over the group dimension with `keepdim=True`, subtract it,
compute the squared deviations, sum over the group dimension, divide by $G$, take the square root,
and divide the centred values by the std plus `eps`. Return a new tensor of shape `(P, G)`.
