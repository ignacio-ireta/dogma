---
title: Stacks and queues
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Pick the structure from the item the answer needs next.

| The answer needs... | Structure | Operations |
|---|---|---|
| the most recent unresolved item | stack (LIFO) | push and pop at the top |
| the oldest unprocessed item | queue (FIFO) | enqueue at the back, dequeue at the front |
| the oldest item and the newest item | deque | both ends |

A **monotonic stack** keeps indices whose values are monotone: for next-greater questions, the values on the
stack never increase from bottom to top. When a new value breaks that order, pop until the order holds again.
Each popped index is resolved by the new index.

Python gives you a list as a stack (`append` and `pop` at the end). For a queue, use `collections.deque`:
`append`, `pop`, `popleft` and `appendleft` are all constant time [Python deque documentation](https://docs.python.org/3/library/collections.html#collections.deque).

## Worked example

Days until a higher reading for `values = [5, 3, 8, 8, 2, 9]`. The stack holds indices, and each pop writes an
answer.

| i | value | pops (index: answer = i - index) | stack after |
|---|---|---|---|
| 0 | 5 | none | [0] |
| 1 | 3 | none | [0, 1] |
| 2 | 8 | 1: 2 - 1 = 1; 0: 2 - 0 = 2 | [2] |
| 3 | 8 | none (8 is not below 8) | [2, 3] |
| 4 | 2 | none | [2, 3, 4] |
| 5 | 9 | 4: 1; 3: 2; 2: 3 | [5] |

Index 5 is left on the stack, so its answer is 0. The result is `[2, 1, 3, 2, 1, 0]`. Index 3 stays on the stack
because the equal 8 does not end its wait.

## Why it works

Take an index $i$ and let $j$ be the first later index with a larger value. Until $j$ arrives, nothing larger has
come after $i$, so $i$ is still on the stack. At $j$, the pop test "top value is smaller than the new value" is
true for $i$, so $i$ is popped at exactly $j$. The answer is therefore $j - i$.

The cost is counted by operations. Each index is pushed once and popped at most once, so the total number of pops is
at most $n$, and the whole pass is $O(n)$ amortised [Cormen et al., chapter 17, amortized analysis]. A quadratic scan
does the same work from scratch for each index, which is why it fails on large inputs.

## Signal vs. noise

- **Signal:** the order the answer needs, the pop test, and the push-and-pop count.
- **Signal:** the front of a sliding-window deque is the answer, and it must be checked for expiry on every step.
- **Noise for now:** whether you use `list` or a class with a stack interface. The operations are what matter.

## Where it breaks

- **`if` instead of `while`.** One new value can resolve several waiting indices. On `[5, 4, 3, 2, 1, 6]`, an `if`
  resolves only index 4, and the other four stay at 0.
- **`>=` in the pop test.** An equal reading ends a wait it should not end. On `[2, 2, 3]` the first 2 should wait
  2 steps.
- **`list.pop(0)` as a queue.** Removing the front shifts every element. Draining $n$ items shifts $(n-1)+(n-2)+\cdots+1=n(n-1)/2$ entries, so total work is quadratic. Draining a deque is linear.
- **Sliding-window deque without expiry.** The front can hold an index that has left the window, so the window
  reports a value that is no longer inside it.
- **Expiry with duplicate values.** Indices make expiry unambiguous. A value-only deque also works if it retains duplicates and removes the front only when it equals the departing value.

## Interview angle

*Answer shape:* name the order the answer needs, choose the structure, state the invariant, then argue O(n) from
pushes and pops. For a window, add the expiry check.

*Follow-up 1: stock span (days of rising or flat price before today)?* Search backwards for the previous strictly
greater price, using a stack. Pop while the top price is at most today's. The span is today's index minus the top
index, or today's index plus one if the stack is empty.

*Follow-up 2: why not a heap for the sliding window maximum?* A heap gives O(n log n), and removing the expired
element needs lazy deletion. A monotonic deque does the same job in O(n). A block method is also O(n), using prefix
and suffix maxima over blocks of width $k$.

## Connections

- **Prerequisites:** [algo/two-pointers-and-sliding-windows], for the window that the deque tracks.
- **Unlocks:** [algo/heaps-and-priority-queues], where the order is by priority instead of by time, and
  [algo/bfs-and-dfs-on-graphs-and-grids], where a queue and a stack are the two frontier strategies.
- **Lineage:** stacks and queues are the first structures in Cormen et al. (chapter 10). The monotone queue for
  window extremes is a standard refinement of the same idea.
