## Hint 1 — concept

Each query has an absolute position, and each key has one too. The queries are the newest tokens, so they sit at the end of the key sequence, not at its start. Decide which keys a query at that position may see, then separately decide which keys are padding.

## Hint 2 — formula / approach

Query i has absolute position S minus L plus i. Query i may attend key j when j is at most that position. This is one comparison over an (L, S) grid. For padding, AND the result with the per-sequence key indicator, broadcast along the query axis. The result stays boolean and fully masked rows need no special case.

## Hint 3 — code shape

Build `i = arange(L)` as a column and `j = arange(S)` as a row, then compare `j <= (S - L) + i`. Validate the sizes and the key_valid type first, then return either the (L, S) grid or its AND with `key_valid.unsqueeze(1)`.
