## Hint 1 — concept

A name can be placed once every name that must come before it has been placed. Ask which names are
ready at the start, and what placing one of them does to the names that depend on it. For the smallest
order, the question at each step is: among the ready names, which one is smallest?

## Hint 2 — formula / approach

Count, for each name, how many prerequisites it still waits for (its in-degree). Names with in-degree 0
are ready. Repeatedly place the smallest ready name, then lower the count of each name that depends on it,
and mark those that reach zero as ready. A min-heap holds the ready names so that the smallest is available
in logarithmic time. If the loop ends with names still unplaced, those names are waiting on each other, which
is a cycle. To name one cycle, start at an unplaced name and step back to an unplaced prerequisite until a
name repeats.

## Hint 3 — code shape

Build two dictionaries first, successors and in-degree, from the edges. Start the heap with the names whose
in-degree is zero, call `heapify`, and loop with `heappop`. Decrement and `heappush` in the inner loop. After
the loop, compare the length of the output with the number of names. For the cycle message, keep the walk as a
list and the position of each name in it, and read the repeated segment in reverse:

```python
prev = min(p for p in predecessors[current] if p not in placed)
```
