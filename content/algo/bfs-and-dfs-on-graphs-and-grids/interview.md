# Interview notes — BFS and DFS on graphs and grids

## Q1. Shortest number of steps through a grid with walls.
**60-second answer.** Each open cell is a node and each move costs one, so breadth-first search gives the shortest
distance. I would use a FIFO queue and mark each cell as visited when I enqueue it, because then its distance is
final the first time it is reached. Walls are never enqueued. The search costs time proportional to the number of
cells. If the goal is unreachable, the answer is minus one.
**Follow-ups.** Why not DFS? It finds a path but not necessarily the nearest one. What if moves had different costs?
BFS is no longer correct; Dijkstra's algorithm replaces the queue with a priority queue.

## Q2. How many islands are in this grid?
**60-second answer.** Scan every cell. Each time I meet an unvisited land cell, I count one island and run a search from
it, marking everything it reaches. That is a flood fill, and either a stack or a queue works. An explicit stack avoids
Python's recursion limit when an island is long and thin.
**Follow-ups.** What if the grid is huge? Time and memory are proportional to the number of cells. Union-find can
count components incrementally when cells arrive as a stream. Diagonal neighbours? Change the neighbour set, and say so.

## Common mistakes
- Marking visited on dequeue, which is correct but does redundant work.
- Using DFS for shortest path.
- Forgetting bounds checks, so a neighbour index goes below zero and wraps around in a list.
- A recursive DFS on a long open region, which raises `RecursionError`.
