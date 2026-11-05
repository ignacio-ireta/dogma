---
title: Heaps and priority queues
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

A binary heap stores items in a complete binary tree, packed into an array. Each parent is no larger than its children,
so the smallest item sits at the root. In `heapq` the array is index 0, the children of index $i$ are $2i+1$ and $2i+2$,
and the parent of $i$ is $\lfloor (i-1)/2\rfloor$.

| Operation | Cost | `heapq` |
|---|---|---|
| Look at the smallest | $O(1)$ | `heap[0]` |
| Insert | $O(\log n)$: sift up | `heappush` |
| Remove the smallest | $O(\log n)$: move the last item to the root, sift down | `heappop` |
| Build from a list of $n$ items | $O(n)$, bottom-up | `heapify` |

`heapq` is a min-heap. For a max-heap, store negated values. Three patterns use it:

- **Top $k$.** Keep a min-heap of the $k$ largest seen. For each item, push it, and pop when the size exceeds $k$. Cost $O(n\log k)$, memory $O(k)$.
- **$k$-way merge.** Keep one head per sorted stream in a heap. Pop the smallest head, emit it, and push the next item from that stream. Cost $O(n\log k)$.
- **Running median.** Keep the lower half in a max-heap and the upper half in a min-heap, with every lower value at most every upper value and sizes differing by at most one. The median is the top of the larger heap, or the mean of both tops.

## Worked example

Merge $[1,4]$ and $[2,4]$. Entries are (key, stream index).

1. Heap: $(1,0),(2,1)$. Pop $(1,0)$ and emit 1. Push the next item of stream 0: $(4,0)$.
2. Heap: $(2,1),(4,0)$. Pop $(2,1)$ and emit 2. Push $(4,1)$.
3. Heap: $(4,0),(4,1)$. The keys tie at 4, and the stream index breaks it, so $(4,0)$ comes first.

Running median of $2,1,3,4$, with the lower heap as a max-heap:

- Add 2: lower $\{2\}$. Median 2.
- Add 1: lower $\{2,1\}$ is too big, so 2 moves up. Lower $\{1\}$, upper $\{2\}$. Median $(1+2)/2=1.5$.
- Add 3: upper $\{2,3\}$ is too big, so 2 moves down. Lower $\{1,2\}$, upper $\{3\}$. Median 2.
- Add 4: upper $\{3,4\}$. Median $(2+3)/2=2.5$.

## Why it works

**Invariant.** Every parent is at most its children. Only the root is ever needed to answer "what is next?", and a sift restores the invariant after one change, along one path of length $\log n$.

**Lazy merge.** A stream is read only when its head has been output, so the merge works on infinite streams. The stream index in each entry breaks ties, so the labels are never compared.

**Median.** The two middle values are the tops of the two halves. Balancing by one move keeps them there. Sorting every prefix would cost $O(n\log n)$ per value, and the heaps cost $O(\log n)$.

## Signal vs. noise

- **Signal:** `heapq` is a min-heap. Negate for a max-heap, and put the ordering key first in each tuple.
- **Signal:** if all the items are known, `heapify` is $O(n)$. Pushing them one by one is $O(n\log n)$.
- **Noise:** heapsort as a sorting algorithm. In practice Python's `sorted` is faster; the heap matters as a tool.

## Where it breaks

- **Comparing payloads.** When two keys tie, Python compares the next field. Two dictionaries cannot be ordered, so the tuple raises `TypeError`. Use an index as the second field.
- **Mutating a stored item.** A heap item changed after the push breaks the invariant without any error.
- **NaN.** NaN compares false with everything, so the heap can misplace it silently. Reject it.
- **Eager merge.** Reading each stream into a list first breaks laziness, and fails on infinite streams.

## Interview angle

"Merge $k$ sorted lists." Answer shape: one head per list in a heap, pop the smallest, push that list's next item. $O(n\log k)$ time, $O(k)$ extra space. For a stream median: two heaps balanced by size, with the middle at the tops, $O(\log n)$ per value. Follow-ups are in `interview.md`.

## Connections

- **Prerequisites:** `algo/sort-then-sweep` (sorted inputs and one pass over them).
- **Unlocks:** priority queues in graph search and schedulers, which come later in the algorithms track.
- **Lineage:** Knuth (vol. 3, section 5.4.1) covers multiway merging; Cormen et al. (chapter 6) cover heaps and priority queues.
- **References:** [Introduction to Algorithms, third edition](https://mitpress.ublish.com/book/introduction-algorithms) covers heaps in chapter 6; the [Python heapq documentation](https://docs.python.org/3.12/library/heapq.html) states the heap invariant and lazy merge behaviour.
