## Hint 1 — concept

A response is scored one token at a time. Each token is predicted by the position just before it,
so the bookkeeping is about which position predicts which token. Only response tokens count toward
the sum.

## Hint 2 — formula / approach

Take log-softmax over the vocabulary at every position. Then the log-probability of token $t$ is the
entry at vocabulary index `input_ids[t]` from the distribution at position $t - 1$. Sum those
entries over response positions for each row. The mask must line up with the token being scored, so
it is shifted by one position in the same way as the ids.

## Hint 3 — code shape

Validate first: rank and shape of the logits, integer ids of the same shape, a boolean mask, no
`True` at position 0, and ids inside the vocabulary. Then compute log-softmax over the last
dimension, slice off the last position of the log-probabilities and the first position of the ids
and mask, gather the actual next-token entries, zero the non-response entries, and sum over the
positions. Return a `(B,)` tensor in the logits dtype.
