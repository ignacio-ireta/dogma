## Hint 1 — concept

Read the stream once, with `enumerate`, and keep only `k` items. The first `k` items fill the
slots. Each later item must survive with probability `k / (i + 1)`, which means drawing one
integer over the first `i + 1` positions and keeping the item only when that integer lands in a
slot.

## Hint 2 — formula / approach

For item index `i >= k`, draw `j = rng.integers(0, i + 1)`. If `j < k`, put the item in slot `j`,
which overwrites the old occupant. Otherwise drop the item. Build no list of the whole stream, and
make no other random draws.

## Hint 3 — code shape

Check `k < 0` before you read anything. Then loop over `enumerate(stream)`. Inside the loop, branch
on `i < k` to append, or to draw and maybe replace. Return the reservoir list at the end. It is
already in slot order, and a short stream leaves it short.
