from __future__ import annotations


def smallest_topological_order(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Return the lexicographically smallest topological order of a dependency graph.

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
    raise NotImplementedError
