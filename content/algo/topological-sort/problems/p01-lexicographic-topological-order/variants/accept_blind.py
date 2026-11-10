from __future__ import annotations

import heapq


def smallest_topological_order(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:  # noqa: PLR0912
    """Return the lexicographically smallest topological order of a dependency graph."""
    if not isinstance(nodes, list):
        raise TypeError("nodes must be a list of strings")
    for name in nodes:
        if not isinstance(name, str):
            raise TypeError("node names must be strings")
    if not isinstance(edges, list):
        raise TypeError("edges must be a list of pairs")

    known = set(nodes)
    if len(known) != len(nodes):
        raise ValueError("nodes contains a repeated name")

    succ: dict[str, set[str]] = {name: set() for name in nodes}
    pred: dict[str, set[str]] = {name: set() for name in nodes}
    for pair in edges:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise TypeError("each edge must be a pair of strings")
        a, b = pair
        if not isinstance(a, str) or not isinstance(b, str):
            raise TypeError("each edge must be a pair of strings")
        if a not in known or b not in known:
            raise ValueError(f"edge ({a!r}, {b!r}) names a node that is not in nodes")
        succ[a].add(b)
        pred[b].add(a)

    indegree = {name: len(pred[name]) for name in nodes}
    heap = [name for name in nodes if indegree[name] == 0]
    heapq.heapify(heap)

    order: list[str] = []
    while heap:
        name = heapq.heappop(heap)
        order.append(name)
        for nxt in succ[name]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(heap, nxt)

    if len(order) < len(nodes):
        raise ValueError("graph has a cycle: " + " -> ".join(_find_cycle(nodes, pred, indegree)))
    return order


def _find_cycle(nodes: list[str], pred: dict[str, set[str]], indegree: dict[str, int]) -> list[str]:
    """Return one cycle among unplaced nodes as a list whose first and last names are equal."""
    unplaced = {name for name in nodes if indegree[name] > 0}
    current = min(unplaced)
    walk = [current]
    position = {current: 0}
    while True:
        current = min(p for p in pred[current] if p in unplaced)
        if current in position:
            cycle_backwards = walk[position[current] :] + [current]  # noqa: RUF005
            return list(reversed(cycle_backwards))
        position[current] = len(walk)
        walk.append(current)
