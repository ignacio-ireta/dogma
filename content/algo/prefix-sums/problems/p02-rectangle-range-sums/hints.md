## Hint 1 — concept

Each rectangle can be expressed using four rectangles anchored at the origin.

## Hint 2 — formula / approach

Store sums of cells strictly above and left of each padded table coordinate.
Subtract two unwanted strips from the lower-right prefix and add their overlap.

## Hint 3 — code shape

Allocate an extra zero row and column. Build each row with a running row total
plus the prefix directly above. Inclusive bottom/right coordinates need a plus one.
