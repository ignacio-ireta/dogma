---
title: "Topological sort"
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01 with a heap as the ready set."
---

## The system

A **directed graph** has nodes and edges $(a, b)$, read "$a$ must come before $b$". It is a
**DAG** (directed acyclic graph) when it has no cycle. A **topological order** of a DAG is a
sequence of all nodes in which every edge $(a, b)$ has $a$ before $b$.

Let $V$ be the nodes, $E$ the edges, and $d(v)$ the **in-degree** of $v$: the number of edges that
end at $v$. A node with $d(v) = 0$ has no unplaced prerequisite, so it is **ready**.

Two standard algorithms:

| Method | Rule | Cycle test |
|---|---|---|
| Kahn (in-degree) | Repeatedly place a ready node, then lower the in-degree of its successors; a successor that reaches 0 becomes ready | Some nodes are never placed |
| DFS post-order | Visit each node after its successors finish, then reverse the finish order | A back edge to a node still on the DFS stack |

This topic uses Kahn's method, because its ready set makes the deterministic choice explicit. Each
edge is handled once: $O(V + E)$ time.

## Worked example

Edges: `a -> c`, `b -> c`, `c -> d`. Nodes `a`, `b`, `c`, `d`. Start with in-degrees
$d(a)=0$, $d(b)=0$, $d(c)=2$, $d(d)=1$. Ready: {a, b}.

| Step | Place | Ready set after the step | In-degrees after the step |
|---|---|---|---|
| 1 | a | {b} | c: 1 |
| 2 | b | {c} | c: 0 |
| 3 | c | {d} | d: 0 |
| 4 | d | {} | |

Order: a, b, c, d. For the **lexicographically smallest** order, keep the ready set in a min-heap.
At step 1 the ready set is {a, b}, and the heap takes `a` first. Compare two free names `b, a` with
no edges, listed in that input order. A FIFO queue places `b` first and gives `b, a`. The heap gives
`a, b`, the smaller order the problem asks for.

**Cycle.** Add `d -> a`. Now $d(a)=1$, $d(b)=0$, $d(c)=2$, $d(d)=1$. Only `b` is ready. After `b`,
`c` still has in-degree 1 from `a`, and `a` waits for `d`, which waits for `c`. Nothing else
becomes ready. Placed: {b}. Unplaced: {a, c, d}, and every unplaced node waits for an unplaced
predecessor, so a cycle exists among them.

## Why it works

Invariant: after each step, the placed nodes form a prefix that is closed under prerequisites, and
the in-degree of an unplaced node counts its edges from unplaced nodes only. Placing a node with
in-degree 0 therefore respects all its incoming edges, because they all come from placed nodes. The
order respects every edge.

If the loop ends with unplaced nodes, every unplaced node has an in-degree above 0, so it has an
unplaced predecessor. Walking backwards through unplaced predecessors must repeat a node, and the
repeated part is a cycle. This is why a cycle is found by counting, not by a visited flag.

The heap changes only the choice among ready nodes. The output is still valid, and the smallest ready
node at each step gives the lexicographically smallest order.

## Signal vs. noise

- **Signal:** the in-degree bookkeeping, the ready set, and the cycle test by unplaced nodes. These
  are the three places where a solution goes wrong.
- **Signal:** the ready set's data structure decides the order. A queue gives a valid order; a heap
  gives the smallest one.
- **Noise for now:** the DFS variant, and strongly connected components (Tarjan). They are useful, but
  the in-degree version answers the interview question.

## Where it breaks

- **Duplicate edges.** Count them consistently. Either drop repeats up front, or count each copy in the
  in-degree and decrement once per copy. Mixing the two leaves a count that never reaches 0, so a
  valid graph looks cyclic.
- **Self-loops.** An edge $(a, a)$ is a cycle of one node. Its in-degree never reaches 0.
- **Isolated nodes.** A node with no edges has in-degree 0, so it must be in the initial ready set.
- **Unknown names.** An edge that names a node not in the node list is an input error, not a new node.
- **Reporting a cycle without naming one.** "There is a cycle" does not help a person fix a build.
  The problem's message is exact: `graph has a cycle: ` followed by one cycle, such as
  `graph has a cycle: a -> b -> a`. Build it from the unplaced nodes.

## Interview angle

Say the in-degree and the ready set first, then the loop. For a cycle, say that the output is short,
then walk back through unplaced prerequisites to name the cycle. If the interviewer asks for the
smallest order, say that the ready set becomes a heap, and that the cost is $O((V+E)\log V)$.

## Connections

Topological order is where a dependency graph is ordered for execution: build tools (`make` targets),
package installs, and spreadsheet recalculation. This curriculum uses the same idea. Its `requires:`
graph must be acyclic, and the content gate checks that with a cycle test. The heap comes from
`algo/heaps-and-priority-queues`, and the in-degree map from `algo/hash-maps-and-sets`.
