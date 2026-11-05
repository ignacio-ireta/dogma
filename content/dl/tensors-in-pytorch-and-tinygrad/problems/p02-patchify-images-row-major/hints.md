## Hint 1 — concept

Patchify is a change of how the same numbers are indexed. Split each spatial axis into a grid index and an offset inside the tile, then move the axes into the order the contract names. No value is computed.

## Hint 2 — formula / approach

Write the index of a pixel as (b, c, gi*p + r, gj*p + s). The output axes are (b, grid row, grid column, c, r, s). A permute to that order, followed by a reshape to (B, N, C*p*p), gives the answer. Check which reshape can return a view of the input.

## Hint 3 — code shape

Validate the arguments first. Reshape to (B, C, H//p, p, W//p, p). Permute to (0, 2, 4, 1, 3, 5). Reshape to (B, N, C*p*p) and clone, so the result owns its memory.
