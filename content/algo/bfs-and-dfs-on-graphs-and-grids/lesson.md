---
title: "BFS and DFS on graphs and grids"
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 by writing the queue loop before the neighbour code."
---

## The system

A grid is a graph. Each open cell is a node, and its up, down, left and right neighbours are its edges. Walls are
missing nodes. Two traversals cover almost every interview question:

| Question | Traversal | Container | Mark a cell when |
|---|---|---|---|
| Shortest number of moves | BFS | FIFO queue | you **enqueue** it |
| Reachable at all? Components? | DFS (or BFS) | stack (or queue) | you visit it |

The rule for BFS is the important one. Because every edge costs 1, BFS explores all cells at distance $d$ before any at
distance $d+1$. The first time a cell is reached, the distance is already final. So mark it visited when you enqueue
it. If you wait until you dequeue it, the same cell can sit in the queue several times, and the work grows with
every extra copy.

A grid of $R \times C$ cells has $RC$ nodes and at most $4RC$ edges. A visited array makes BFS cost
$O(RC)$ time and memory.

## Worked example

Start $S$ at row 0, column 0. Goal $G$ at row 2, column 2. `#` is a wall.

```text
S . .
# # .
. . G
```

Level by level from $S$, with distance in brackets:

- Level 0: $S$ [0].
- Level 1: (0,1) [1]. The cell (1,0) is a wall, so it does not enqueue.
- Level 2: (0,2) [2].
- Level 3: (1,2) [3].
- Level 4: (2,2) [4], which is $G$.

The answer is 4 moves. DFS on the same grid can also find a path of 4 here, if it tries right first. In general a DFS path can
be far longer than the shortest one, because it commits to a direction and does not return until it hits a dead end.

For components, scan every open cell. Start a search at each unvisited open cell, count one component, and mark
everything reachable from it. Two open cells that touch by a side belong to the same component.

## Why it works

**Shortest paths.** Levels grow by one. Every cell at distance $d$ enqueues its unvisited neighbours at $d+1$. The
queue is FIFO, so distances never decrease along it, and a cell's first enqueue fixes its distance. Each neighbour
reached first, from a cell at distance $d$, has distance $d+1$.

**Why not recursion for DFS.** A recursive DFS on a snake-shaped open region of 50,000 cells needs 50,000 nested calls.
CPython's default recursion limit is 1,000. An explicit stack holds the same frontier as a list and avoids the call
stack. The traversal order changes, but reachability and component counts do not.

## Signal vs. noise

- **Signal:** the question decides the traversal. A "how many moves" question wants BFS. "Is it connected" does not.
- **Signal:** the visited rule. Visited on enqueue for BFS, visited when the stack pops or pushes for DFS, with the
  same result for reachability.
- **Noise:** the neighbour order. The four directions can come in any order without changing distances.

## Where it breaks

- **No visited set.** The search follows cycles forever, or, with a depth cap, explores exponentially many routes
  to the same cells.
- **DFS for shortest path.** It returns some path, not the shortest.
- **Visit-on-dequeue.** Still correct, because the first dequeue of a cell also carries its final distance. It is slower than visit-on-enqueue, since a cell can sit in the queue several times, which costs time and memory.
- **Recursion depth.** A recursive DFS on a long open corridor raises `RecursionError` once the path passes the limit.
- **Start or goal on a wall.** Decide what this means. The problem states it, and a correct search returns the stated answer.

## Interview angle

**60-second answer.** For shortest moves in an unweighted grid I would use BFS with a queue, marking cells on enqueue,
so each cell is queued once and its distance is final when it is reached. Walls are not enqueued. The cost is
$O(RC)$. For components I would scan the grid and start an iterative DFS at each unvisited open cell, using an
explicit stack to avoid the recursion limit.

**Likely follow-ups.** Why not DFS for the shortest path? It finds a path, but it does not stop at the nearest goal.
Why mark on enqueue? Marking on dequeue lets the same cell enter the queue several times.

## Connections

This topic needs the stack and queue containers (`algo/stacks-and-queues`) and fast visited sets (`algo/hash-maps-and-sets`,
or a flat boolean array for grids). It leads to topological sort, which is a DFS with a post-order rule on a directed
graph, and to weighted shortest paths, where BFS stops being correct once edges have different costs.
