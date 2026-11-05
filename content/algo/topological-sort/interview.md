# Interview notes — Topological sort

## Q1. Give a valid build order for these tasks, and say what happens when there is a cycle.
**60-second answer.** I build the in-degree of each task from its prerequisites. The ready set starts
with the tasks whose in-degree is 0. I place a ready task, lower the in-degree of each task that depends
on it, and make the ones that reach 0 ready. If every task is placed, the list is a valid order. If not,
the leftover tasks each wait for another leftover task, so there is a cycle. I report the cycle by walking
back through leftover prerequisites until a task repeats. The whole thing is O(V + E).
**Follow-ups.** Why not a visited flag? In a DFS, a visited flag cannot tell a finished node from one on the
current path, so it reports false cycles. Kahn's count avoids that. Which order do you get? Any valid order;
for a fixed one, use a heap for the ready set.

## Q2. Make the order deterministic, so that the same input always gives the same output.
**60-second answer.** Keep the ready set in a min-heap keyed by the name. The output is then the
lexicographically smallest valid order. The cost goes from O(V + E) to O((V + E) log V), because each push and
pop is logarithmic. A queue is the cheaper choice when any valid order will do.
**Follow-ups.** Is the greedy choice always right? For lexicographic order, yes: placing a ready name never
makes another ready name unready, so the smallest ready name can always go first. Are there ties? Names are
unique, so the heap never has to break one.

## Q3. How would you use this in a real system?
**60-second answer.** A build tool orders targets by their dependencies and rejects cycles with a message.
A package manager installs dependencies before their dependents. A notebook or spreadsheet recalculates cells in
dependency order. In each case the cycle error is the useful output: it tells the user which edge to remove.
**Follow-ups.** What if the graph changes while you run? Recompute the in-degrees for the affected part, or rebuild.
What if the graph is huge? The algorithm is linear, so the cost is reading the edges, not the sort.

## Common mistakes
- Forgetting isolated nodes, which have in-degree 0 and must start in the ready set.
- Counting duplicate edges in the in-degree but following each only once, so a valid graph looks cyclic.
- Reporting "cycle" without naming one.
- Using a plain queue and calling the result the smallest order.
- Missing a self-loop, which is a cycle of one node.
