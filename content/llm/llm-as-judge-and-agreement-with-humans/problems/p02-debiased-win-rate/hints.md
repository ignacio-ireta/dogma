## Hint 1 — concept

A verdict names a position, not a candidate. Once you know which candidate sat in which position in
each order, you can translate both verdicts into statements about `X`. Only an item where `X` wins
both translated verdicts is a full win for `X`.

## Hint 2 — formula / approach

Translate the forward verdict with `X` in position A, and the reverse verdict with `X` in position
B. Each item scores `1.0` if both translations say `X`, `0.0` if both say `Y`, and `0.5` for every
other combination. The win rate is the mean of those scores. Remember that a tie in either order
never produces a full win.

## Hint 3 — code shape

Validate the input once: non-empty, each element a pair of strings from the verdict set. Then loop
over the items. For each item, map the two verdicts to a candidate label with a small helper, and
let an if-else chain decide between `1.0`, `0.0` and `0.5`. Accumulate the total, and divide by the
number of items at the end.
