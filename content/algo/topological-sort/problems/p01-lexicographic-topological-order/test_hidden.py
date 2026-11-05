"""Hidden tests: a brute-force search over permutations on small graphs (acyclic and
cyclic), larger random DAGs checked edge by edge, duplicate and isolated names, self-loops,
invalid input, and non-mutation.

The brute-force oracle tries permutations of the sorted names in order, so the first valid
one is the lexicographically smallest. Learner code is imported from submission.py. The
tests do not use graphlib or any solution file.
"""

from __future__ import annotations

import itertools
import random
from itertools import pairwise

import pytest
from submission import smallest_topological_order


def _brute(nodes, edges):
    """First valid permutation of the sorted names, or None when the graph has a cycle."""
    for perm in itertools.permutations(sorted(nodes)):
        position = {name: i for i, name in enumerate(perm)}
        if all(position[a] < position[b] for a, b in edges):
            return list(perm)
    return None


def _order(nodes, edges):
    message, failure = None, None
    try:
        result = smallest_topological_order(nodes, edges)
    except ValueError as exc:
        message = str(exc)
    except Exception as exc:
        failure = f"your function raised {type(exc).__name__} on a valid graph: {exc}"
    else:
        failure = None
    if message is not None:
        failure = (
            "a valid graph raised ValueError: a cycle is reported only when names stay unplaced; "
            f"check the in-degree counts, including repeated edges ({message})"
        )
    if failure is not None:
        pytest.fail(failure)
    if type(result) is not list:
        pytest.fail("return a list of node names, not a tuple, a generator or a set")
    return result


CYCLE_PREFIX = "graph has a cycle: "


def _cycle_failure(message, edges):
    """Return why a cycle message is wrong, or None when it names a real cycle."""
    if not message.startswith(CYCLE_PREFIX):
        return (
            "a cycle error must read 'graph has a cycle: a -> b -> a': the exact prefix, then the "
            "names joined by ' -> '"
        )
    names = [part.strip() for part in message[len(CYCLE_PREFIX) :].split("->")]
    if len(names) < 2 or names[0] != names[-1]:
        return "the named cycle must start and end with the same name"
    edge_set = set(edges)
    for step in pairwise(names):
        if step not in edge_set:
            return "the named cycle uses a step that is not an edge of the graph"
    return None


def _expect_cycle(nodes, edges):
    message, failure = None, None
    try:
        smallest_topological_order(nodes, edges)
    except ValueError as exc:
        message = str(exc)
    except Exception as exc:
        failure = f"a graph with a cycle must raise ValueError, instead {type(exc).__name__}"
    else:
        failure = (
            "a cyclic graph was accepted: some names were never placed, so check that the output "
            "covers every name before returning"
        )
    if message is not None:
        failure = _cycle_failure(message, edges)
    if failure is not None:
        pytest.fail(failure)


def _expect_cycle_free(nodes, edges):
    expected = _brute(nodes, edges)
    got = _order(nodes, edges)
    if got != expected:
        pytest.fail(
            "a graph without a cycle returned the wrong order: check the ready set and the counts"
        )


def _random_dag(rng, names, density):
    """Edges that follow a hidden ranking, so the graph is acyclic by construction."""
    ranking = names[:]
    rng.shuffle(ranking)
    edges = []
    for i, a in enumerate(ranking):
        for b in ranking[i + 1 :]:
            if rng.random() < density:
                edges.append((a, b))
    rng.shuffle(edges)
    return edges


def test_matches_brute_force_on_seeded_small_dags():
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic graphs, not security.
    for _ in range(200):
        names = [f"n{i}" for i in range(rng.randrange(0, 8))]
        nodes = names[:]
        rng.shuffle(nodes)
        edges = _random_dag(rng, names, rng.random())
        expected = _brute(nodes, edges)
        got = _order(nodes, edges)
        if got != expected:
            pytest.fail(
                "the order differs from the lexicographically smallest valid order: "
                "at each step take the smallest ready name; keep the ready set in a heap"
            )


def test_small_graphs_with_cycles_raise_value_error_naming_a_cycle():
    rng = random.Random(7)  # noqa: S311 -- seeded synthetic graphs.
    for _ in range(150):
        names = [f"c{i}" for i in range(rng.randrange(1, 7))]
        edges = [(a, b) for a in names for b in names if rng.random() < 0.25]
        if _brute(names, edges) is None:
            _expect_cycle(names, edges)
        else:
            _expect_cycle_free(names, edges)


def test_larger_random_dag_respects_every_edge_and_places_every_name():
    rng = random.Random(99)  # noqa: S311 -- seeded synthetic graph.
    names = [f"v{i:04d}" for i in range(2000)]
    nodes = names[:]
    rng.shuffle(nodes)
    edges = []
    for _ in range(6000):
        i, j = rng.randrange(2000), rng.randrange(2000)
        if i < j:
            edges.append((names[i], names[j]))
    rng.shuffle(edges)
    result = _order(nodes, edges)
    if sorted(result) != sorted(names):
        pytest.fail("every name must appear exactly once in the order: check the placement loop")
    position = {name: i for i, name in enumerate(result)}
    for a, b in edges:
        if position[a] >= position[b]:
            pytest.fail(f"an edge is violated: {a} must come before {b}")


def test_duplicate_edges_change_nothing():
    nodes = ["d", "b", "a", "c"]
    base = [("a", "b"), ("c", "b"), ("b", "d")]
    doubled = base + base + [("a", "b")]
    assert _order(nodes, doubled) == _order(nodes, base), (
        "duplicate edges must not change the order or report a cycle: count each prerequisite "
        "once per edge and decrement once per edge"
    )


def test_isolated_names_are_included_and_ordered_with_the_free_names():
    got = _order(["q", "b", "k"], [("k", "b")])
    assert got == ["k", "b", "q"], (
        "names with no edges are free from the start and must appear in the order; here k comes "
        "first, then b, then the free name q"
    )


def test_self_loop_is_a_cycle_named_with_its_node():
    try:
        smallest_topological_order(["a", "b"], [("a", "a"), ("a", "b")])
    except ValueError as exc:
        assert "a -> a" in str(exc), (
            "a self-loop is a cycle of one name: the message must read 'a -> a'"
        )
        return
    pytest.fail("a self-loop was accepted: an edge from a name to itself is a cycle")


def test_unknown_name_in_an_edge_raises_value_error_not_key_error():
    with pytest.raises(ValueError):
        smallest_topological_order(["a", "b"], [("a", "zz")])


def test_duplicate_node_name_raises_value_error():
    with pytest.raises(ValueError):
        smallest_topological_order(["a", "a", "b"], [])


def test_non_string_name_raises_type_error():
    with pytest.raises(TypeError):
        smallest_topological_order([1, 2], [])


def test_long_chain_of_names_comes_out_in_dependency_order():
    names = [f"s{i:04d}" for i in range(1000)]
    shuffled = names[:]
    random.Random(3).shuffle(shuffled)  # noqa: S311 -- seeded shuffle.
    edges = [(names[i], names[i + 1]) for i in range(999)]
    assert _order(shuffled, edges) == names, (
        "a chain of 1000 names must come out in chain order, whatever order the nodes are listed in"
    )


def test_inputs_are_not_mutated():
    nodes = ["c", "a", "b"]
    edges = [("c", "a"), ("b", "c")]
    nodes_before = list(nodes)
    edges_before = list(edges)
    _order(nodes, edges)
    if nodes != nodes_before or edges != edges_before:
        pytest.fail("the function changed its inputs: sort a copy, not nodes or edges")


def test_nodes_that_is_not_a_list_raises_type_error():
    try:
        smallest_topological_order(("a", "b"), [])
    except TypeError:
        return
    except Exception as exc:
        pytest.fail(f"nodes given as a tuple raised {type(exc).__name__}, not TypeError")
    pytest.fail("nodes must be a list: a tuple of names must raise TypeError")


def test_edge_that_is_a_three_item_tuple_raises_type_error():
    try:
        smallest_topological_order(["a", "b", "c"], [("a", "b", "c")])
    except TypeError:
        return
    except Exception as exc:
        pytest.fail(f"an edge with three items raised {type(exc).__name__}, not TypeError")
    pytest.fail("each edge must be a pair of node names: a three-item edge must raise TypeError")


def test_edge_given_as_a_string_raises_type_error():
    try:
        smallest_topological_order(["a", "b"], ["ab"])
    except TypeError:
        return
    except Exception as exc:
        pytest.fail(f"an edge written as a string raised {type(exc).__name__}, not TypeError")
    pytest.fail(
        "each edge must be a pair of node names: a string is not a pair and must raise TypeError"
    )


def test_edge_with_a_non_string_end_raises_type_error():
    try:
        smallest_topological_order(["a", "b"], [(1, "b")])
    except TypeError:
        return
    except Exception as exc:
        pytest.fail(f"an edge with a non-string end raised {type(exc).__name__}, not TypeError")
    pytest.fail("each edge must be a pair of strings: a non-string end must raise TypeError")
