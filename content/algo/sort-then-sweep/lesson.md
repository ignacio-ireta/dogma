---
title: Sort, then sweep
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01."
---

## The system

Sorting first turns a question about pairs into a question about neighbours. Two patterns cover most cases.

| Question | Sort key | Rule at a tie |
|---|---|---|
| merge closed intervals `[start, end]` | `start` | touching intervals merge, because they share a point |
| peak overlap of half-open sessions `[start, end)` | time, then change | an end (-1) comes before a start (+1) at the same time |

**Merge pattern.** Walk the sorted intervals, and keep the last merged interval. A new interval merges when
`start <= last_end`, and the merged end is `max(last_end, end)`. Otherwise it starts a new group.

**Sweep pattern.** Turn each session into a start event with +1 and an end event with -1. Sort the events, keep a
running total, and take its maximum. After all events at one time are processed, the total is the number of active sessions at that time. Intermediate totals at ties can be smaller; ends first prevent a false peak.

## Worked example

Merge `[(5, 7), (1, 2), (2, 4), (9, 9)]`. Sorted by start: `(1, 2), (2, 4), (5, 7), (9, 9)`.

| interval | test | result |
|---|---|---|
| (1, 2) | no last group | `[(1, 2)]` |
| (2, 4) | 2 <= 2, touching | `[(1, 4)]` |
| (5, 7) | 5 <= 4? no | `[(1, 4), (5, 7)]` |
| (9, 9) | 9 <= 7? no | `[(1, 4), (5, 7), (9, 9)]` |

Peak overlap of `(1, 3), (3, 5), (5, 7)`. Events sorted with ends first at a tie:
`(1,+1), (3,-1), (3,+1), (5,-1), (5,+1), (7,-1)`. The running totals are 1, 0, 1, 0, 1, 0, so the peak is 1. If
starts came first at 3 and at 5, the totals would be 1, 2, 1, 2, 1, 0, and the peak would be a false 2.

## Why it works

After sorting by start, every earlier interval starts no later than the current one. If the current start is larger
than the merged block's end, then every interval in the block ends before it, so nothing earlier can share a point with
it. The block is finished, and one pass is enough. For the sweep, the running total at any moment equals the number of
sessions that contain that moment. The maximum is reached right after a start, and ends are applied first at a tie,
which is the half-open rule: a session that ends at $t$ is not active at $t$. Sorting costs $O(n \log n)$. Comparison
sorts cannot do better in general [Cormen et al., chapter 8], so the pass after the sort adds only $O(n)$.

## Signal vs. noise

- **Signal:** the sort key, the tie rule, and `max` for the merged end.
- **Signal:** closed intervals merge on a shared point, and half-open sessions do not overlap on a shared endpoint.
- **Noise for now:** the stability of the sort beyond the key, and how the tuples are laid out in memory.

## Where it breaks

- **No sort.** The pass compares only with the last group in input order. On `[(5, 7), (1, 2), (2, 4)]` it merges (1, 2) into (5, 7) and returns `[(5, 7)]`, not `[(1, 4), (5, 7)]`.
- **Sorting by end.** `[(5, 6), (0, 10), (1, 2)]` sorted by end puts `(0, 10)` last, and the answer splits into
  `(1, 2)` and `(5, 10)`. The long interval must come first.
- **Taking the newest end.** `(0, 10)` followed by `(2, 3)` gives `(0, 3)` if the end is replaced. Use `max`.
- **`start <= last_end + 1`.** On integers, `(1, 3)` and `(4, 5)` would merge, but they share no point.
- **Starts before ends at a tie.** `(1, 3)` and `(3, 5)` count as two active sessions at 3, so the peak is wrong.
- **Pairwise comparison.** Checking every pair, or repeating a pass until nothing changes, is quadratic. A 200,000-item
  list does not finish in the time limit.

## Interview angle

*Answer shape:* name the sort key, the state you keep during the pass, the tie rule, then the complexity:
one sort, O(n log n), and one pass, O(n).

*Follow-up 1: minimum number of meeting rooms?* It equals the peak overlap of the meetings, found by the sweep.
Meetings that end at the moment another starts can share a room, so the tie rule matters.

*Follow-up 2: why not compare all pairs?* That costs O(n squared). After sorting, an interval can only interact
with the group that is still open, so one comparison per interval is enough.

## Connections

- **Prerequisites:** [algo/binary-search], because sorted data is what makes searching and sweeping possible.
- **Unlocks:** greedy interval scheduling, and sweep lines in geometry. A heap of end times gives the room
  assignment version.
- **Lineage:** sorting as preprocessing is one of the standard algorithm-design recipes [Skiena, chapter 4].
