from __future__ import annotations

import heapq


def smallest_topological_order(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Wrong on purpose: a self-loop (a, a) is ignored instead of reported as a cycle.

    Return the lexicographically smallest topological order of a dependency graph.

    Parameters
    ----------
    nodes : list[str]
        Distinct node names. An empty list is allowed. Must not be mutated.
    edges : list[tuple[str, str]]
        Pairs (a, b) meaning a must come before b. Each name must be in nodes. Duplicate edges are
        allowed. Must not be mutated.

    Returns
    -------
    list[str]
        Every name in nodes exactly once, respecting every edge, and lexicographically smallest
        among all such orders.

    Raises
    ------
    ValueError
        When nodes repeats a name, an edge names an unknown node, or the graph has a cycle. The
        message for a cycle is "graph has a cycle: a -> b -> ... -> a".
    TypeError
        When nodes is not a list, or a name or an edge is not made of strings.
    """
    successors, predecessors, indegree = _build_graph(nodes, edges)

    # Step 2 (the ready set): a min-heap of the names whose prerequisites are all placed. Taking
    # the smallest ready name at each step gives the lexicographically smallest order. A plain
    # queue would take names in the order they became ready, which is not the smallest order.
    ready = [name for name in nodes if indegree[name] == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        name = heapq.heappop(ready)
        order.append(name)
        # Placing name satisfies one prerequisite of each successor. A successor whose count
        # reaches zero is ready now.
        for nxt in successors[name]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(ready, nxt)

    # Step 3 (cycles): a name that was never placed still waits for a prerequisite that is also
    # never placed. Walking backwards through unplaced prerequisites must therefore repeat a name,
    # and the repeated part is a cycle.
    if len(order) < len(nodes):
        raise ValueError(
            "graph has a cycle: " + " -> ".join(_find_cycle(nodes, order, predecessors))
        )
    return order


def _build_graph(
    nodes: list[str], edges: list[tuple[str, str]]
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, int]]:
    """Validate the input and return successors, predecessors and in-degrees."""
    if not isinstance(nodes, list):
        raise TypeError("nodes must be a list of strings")
    if any(type(name) is not str for name in nodes):
        raise TypeError("every node name must be a str")
    known = set(nodes)
    if len(known) != len(nodes):
        raise ValueError("duplicate node name in nodes")

    # indegree[b] counts the edges into b: the number of prerequisites b still waits for.
    # Duplicate edges are counted each time and also followed each time, so the counts agree.
    successors: dict[str, list[str]] = {name: [] for name in nodes}
    predecessors: dict[str, list[str]] = {name: [] for name in nodes}
    indegree = dict.fromkeys(nodes, 0)
    for pair in edges:
        if not (isinstance(pair, tuple | list) and len(pair) == 2):
            raise TypeError("each edge must be a pair of node names")
        a, b = pair
        if type(a) is not str or type(b) is not str:
            raise TypeError("each edge must be a pair of node names")
        if a not in known or b not in known:
            raise ValueError(f"edge {a!r} -> {b!r} names a node that is not in nodes")
        if a == b:
            continue
        successors[a].append(b)
        predecessors[b].append(a)
        indegree[b] += 1
    return successors, predecessors, indegree


def _find_cycle(
    nodes: list[str], order: list[str], predecessors: dict[str, list[str]]
) -> list[str]:
    """Return one cycle among the names missing from order, as a closed list of names."""
    placed = set(order)
    start = min(name for name in nodes if name not in placed)
    path = [
        start
    ]  # path[k + 1] is a prerequisite of path[k], so the edges run path[k+1] -> path[k]
    position = {start: 0}
    while True:
        current = path[-1]
        # Every unplaced name has an unplaced prerequisite; take the smallest for a fixed output.
        prev = min(p for p in predecessors[current] if p not in placed)
        if prev in position:
            # The walk closed at prev = path[i]. Reading the segment forwards gives the cycle.
            cycle = list(reversed(path[position[prev] :]))
            return [*cycle, cycle[0]]
        position[prev] = len(path)
        path.append(prev)
