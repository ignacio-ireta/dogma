# Lexicographically smallest topological order

A set of modules must be built in an order that respects their dependencies. An edge `(a, b)`
means that `a` must come before `b`. Implement `smallest_topological_order(nodes, edges)`, which returns
every node name in such an order. Among all valid orders, it must return the lexicographically smallest:
the first position gets the smallest name that can go first, then the smallest name that can go next, and
so on.

If the dependencies form a cycle, no valid order exists. Raise `ValueError` and name one cycle, so
the person can see where to look.

## Contract

- `nodes`: a `list` of distinct `str` names. An empty list is allowed.
- `edges`: a `list` of `(a, b)` pairs of `str`. Each name must be in `nodes`. Duplicate edges are
  allowed and change nothing. A list or tuple of two strings is accepted for each pair.
- Returns: a new `list` of `str` with every name in `nodes` exactly once. For every edge `(a, b)`, `a`
  appears before `b`. Among all such lists it is the lexicographically smallest, comparing names with
  Python's `str` ordering.
- Raises `ValueError` when `nodes` has a repeated name, when an edge names a node that is not in `nodes`,
  or when the dependencies contain a cycle. A self-loop `(a, a)` is a cycle.
- For a cycle, the message is exactly `graph has a cycle: ` (with the trailing space), followed by the names
  of one cycle joined by ` -> `. The first and last names are equal, and every step is an edge of the graph.
  For example: `graph has a cycle: b -> c -> a -> b`. A self-loop `(a, a)` gives `graph has a cycle: a -> a`.
- Raises `TypeError` when `nodes` is not a list, or when a name or an edge is not a string pair.
- Must not mutate `nodes` or `edges`.
- Time: O((V + E) log V) for V nodes and E edges.

## Examples

```python
smallest_topological_order(["m", "z", "a"], [("z", "a")])
# ["m", "z", "a"]      m and z can go first; m is smaller. a waits for z.

smallest_topological_order(["b", "a", "c"], [("b", "c")])
# ["a", "b", "c"]      a is free and smaller than b, so it goes first.
#                      (A queue that takes b first would give ["b", "a", "c"], which is valid but not smallest.)

smallest_topological_order(["x", "y"], [("x", "y"), ("y", "x")])
# ValueError: graph has a cycle: x -> y -> x
```

```python
smallest_topological_order([], [])
# []
```

## Constraints

- Allowed: built-in types and the standard library. A `heapq` heap is the intended tool for the
  ready names.
- Not allowed: `graphlib`, which the standard library provides for topological sorting. Writing the sort is
  the point of this problem.

## Done looks like

Your function returns the lexicographically smallest valid order on the examples, and it names a real
cycle when one exists. The hidden tests compare with a brute-force search over permutations on small
graphs, both acyclic and cyclic. They check that every edge is respected on larger random DAGs, that duplicate
edges change nothing, that isolated names are included, and that a self-loop is a cycle. They also check
that the inputs are not mutated, the type rules, and that the message for a cycle names a real cycle.
