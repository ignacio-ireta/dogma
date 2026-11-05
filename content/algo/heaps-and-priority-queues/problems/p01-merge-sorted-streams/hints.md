## Hint 1 — concept

A k-way merge keeps one candidate from each sorted stream: the head, the smallest pair not yet taken. The
smallest of the k heads is the next output. A heap finds that minimum in O(log k), so the merge costs
O(log k) per output, not O(k). Reading lazily means a stream is read only when its head has been output.

## Hint 2 — formula / approach

Read the first pair of each stream and push `(key, stream index, label)` onto a heap. Then, in a loop, pop
the smallest entry, yield its key and label, read the next pair from the same stream, and push it. The
stream index in the tuple does two jobs: it breaks ties in favour of the lower stream, and it keeps labels
out of every comparison, because two entries never have the same index.

## Hint 3 — code shape

Make the function a generator: `iter()` each stream first, then prime the heap with `next(iterator,
sentinel)` so an empty stream is not an error. Use `heapq.heappush` and `heapq.heappop` only. In the loop,
pop, `yield`, then refill that stream. Do not call `list()` on a stream, and do not sort the whole input.
