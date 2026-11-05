# Interview notes: heaps and priority queues

## Q1. Merge k sorted lists into one sorted sequence.

**60-second answer.** I keep one head from each list in a min-heap, with the list index as a tie-breaker. I pop the
smallest head, emit it, and push the next item from the same list. That is O(n log k) time for n items, and O(k) extra
space. Because I read each list only when its head is output, the merge also works on streams that never end. A
concatenate-and-sort gives the same answer in O(n log n) but reads everything first, and it cannot be lazy.

**Follow-ups.**
- *Why the index in the heap entry?* It breaks ties in favour of the lower list, and it keeps the payload out of every comparison.
- *What about k = 1 million lists?* Use a heap of stream heads, which is still O(log k) per output; the cost per item grows only logarithmically.

## Q2. Given a stream of numbers, report the median after each one.

**60-second answer.** I keep the lower half in a max-heap, stored as negatives, and the upper half in a min-heap. I push
each value to the correct half, then move one item across if the sizes differ by more than one. The median is the top
of the larger heap, or the mean of both tops when the sizes are equal. Each value costs O(log n). Sorting the prefix
every time costs O(n log n) per value, which is too slow for long streams.

**Follow-ups.**
- *Why do the two heaps stay balanced?* Every insert is followed by at most one move, so the sizes never differ by more than one, and the middle values stay at the tops.
- *How would you handle removals?* Use lazy deletion: mark the value and skip it when it reaches a top, keeping a count of the removed items.

## Common mistakes

- Using a Python min-heap as a max-heap without negating the values.
- Comparing payloads on equal keys, which raises `TypeError` for objects that cannot be ordered.
- Re-sorting the whole input after each value, which is correct but O(n² log n) for a long stream.
- Accepting NaN, which has no place in an order and can corrupt the heap silently.
