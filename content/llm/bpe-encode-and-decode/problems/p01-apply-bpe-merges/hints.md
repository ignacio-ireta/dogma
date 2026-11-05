## Hint 1 — concept

The merge list is a priority list. Each merge says "join this pair", and the earlier in the list the higher the
priority. Applying a merge can create a new symbol, and that new symbol can form a pair with a neighbour that
has a higher priority than anything else. So the question at each step is which available merge is the most
important, not which pair comes first in the word. Think about what happens to `abc` when `(b, c)` is learned
before `(a, b)`.

## Hint 2 — formula / approach

Give each merge its rank, which is its index in the list. Build a dict from pair to rank once, in the
constructor. To encode, start from the characters and the end-of-word marker. Repeat: list the adjacent pairs
that are in the dict, take the one with the smallest rank, merge every occurrence of it from left to right, and
stop when no adjacent pair is in the dict. Decoding is the reverse check: join the tokens, require the marker at
the end, and remove it.

## Hint 3 — code shape

1. In `__init__`, validate the merges, then build a dictionary from each pair to its rank (its
   position in the list), so a rank lookup is one step.
2. In `encode`, start from the word's characters followed by the end-of-word symbol.
3. Repeat: collect the adjacent pairs that have a rank. If there are none, stop. Otherwise pick
   the pair with the lowest rank.
4. Rebuild the symbol list in one left-to-right scan: where the chosen pair starts, append the
   joined symbol and move two places; elsewhere copy one symbol and move one place.
5. `decode` joins the tokens, checks the end-of-word symbol is where the contract says, and
   removes it.

Test your loop on `abc` with the merges `[("b", "c"), ("a", "b")]` by hand. You should get
`['a', 'bc', '</w>']`.
