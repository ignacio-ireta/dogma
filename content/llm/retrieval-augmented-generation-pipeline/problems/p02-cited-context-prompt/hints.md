## Hint 1 — concept

Retrieval has two separate decisions. First choose which chunks are most similar to the query, using a
score that ignores vector length. Then decide which of those candidates fit into the word budget,
using a rule that does not stop at the first chunk that is too long.

## Hint 2 — formula / approach

The cosine is `dot(q, v) / (norm(q) * norm(v))`. Compute it for every chunk, round it to 9 decimals so
near-equal scores tie, and sort by `(-score, chunk id)`. Keep the first `min(k, m)` of that order.
Then walk those candidates in order, adding each one whose word count still fits in the remaining
budget. Build the output from the selected chunks only.

## Hint 3 — code shape

Validate every argument before any arithmetic. Compute one array of cosines with a single matrix
product divided by the norms, then build a list of rounded scores. Sort the indices with a key tuple.
Pack with a running `used` total and a `selected` list. Finally join the lines: the `Context:` header,
the `[id] text` lines, a blank line and the question.
