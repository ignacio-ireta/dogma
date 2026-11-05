## Hint 1 — concept

Each sampled token is a choice the policy could have made differently. The ratio measures how much
the policy has moved on that token since sampling, and the clip stops the update from pushing one
token too far. The advantage gives the direction, and the reference term keeps the policy anchored.

## Hint 2 — formula / approach

Per token: $\rho = \exp(\text{logp\_new} - \text{logp\_old})$. The surrogate is the minimum of
$\rho A$ and $\operatorname{clip}(\rho, 1-\varepsilon, 1+\varepsilon) A$, with $A$ taken from the
response. The $k_3$ term uses $\delta = \text{logp\_ref} - \text{logp\_new}$ and equals $e^{\delta} - \delta - 1$.
Average the masked per-token losses within each response, then across responses.

## Hint 3 — code shape

Validate first: shapes, boolean mask with at least one True per row, matching floating dtypes, finite
values, a clip width strictly between 0 and 1, and a non-negative KL coefficient. Then compute the
ratio and the clipped and unclipped surrogates on the whole (G, T) grid, with the advantage broadcast
down each row. Build the $k_3$ term and the per-token loss. Zero out the masked positions with `where`,
sum each row, divide by its mask count, and average over rows.
