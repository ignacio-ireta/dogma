# Interview notes — Causal masking

## Q1. Build the mask for a decoding step where two new queries attend a four-token cache.

**60-second answer.** The queries are the last two positions, so the offset is S minus L, which is 2. Query i may attend key j when j is at most 2 plus i. The rows are [1,1,1,0] and [1,1,1,1]. Padding is a per-sequence key mask that is ANDed in along the key axis. I would test this case explicitly, because the common shortcut of a square lower triangle is wrong here.

**Follow-ups.** Why not is_causal? It is top-left aligned for non-square masks, so the first query would see only the first key. What changes for training? L equals S and the offset is zero, so the mask is the plain lower triangle.

## Q2. A row has no allowed key. What happens and what should it return?

**60-second answer.** The row maximum is minus infinity, so subtracting it gives NaN, and the softmax output is NaN. NaN spreads to every later layer. The fix is an explicit policy: return zero weights and a zero output for that row, and test that no NaN appears. The policy should be stated in the function's contract, not left to whatever the framework does.

**Follow-ups.** Is zero the only choice? It is the simplest that keeps the output finite; the alternative is to raise an error, which is right when a fully masked row means a bug. Does this happen in practice? Yes, for padded sequences and for queries that precede every key.

## Common mistakes

- Using a top-left triangle when the queries are the newest tokens of a cache.
- Applying the padding mask along the query axis instead of the key axis.
- Letting a fully masked row produce NaN and only noticing it in the loss.
