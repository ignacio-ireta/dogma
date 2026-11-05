## Hint 1 — concept

A vote needs two things before it can count: answers that mean the same thing must be
recognised as the same, and every tie must be settled by a rule you wrote down in advance. Blank
answers are not votes at all.

## Hint 2 — formula / approach

Normalise each answer by collapsing whitespace runs, trimming, and lowercasing; skip the empty
results. Count each normalised answer. Record the index where each answer first appears. The winner
has the largest count, and among equal counts the smallest first index.

## Hint 3 — code shape

Validate the types first: reject a bare string and any non-string element. Loop once over the
answers with `enumerate`, keeping two dictionaries, one for counts and one for first positions, using
`setdefault` for the first position. If both are empty, raise. Otherwise return the minimum over the
counts keys with a key that sorts by negative count and then by first position.
