"""Public tests: the examples from problem.md, a named cycle, and the empty graph.

The learner's code is imported from submission.py. The cycle test checks that the message names a
real cycle, using the edges the test itself passed in.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
from submission import smallest_topological_order


def _order(nodes, edges):
    try:
        result = smallest_topological_order(nodes, edges)
    except ValueError as exc:
        pytest.fail(
            "a valid graph raised ValueError: a cycle is reported only when names stay unplaced; "
            f"check the in-degree counts, including repeated edges ({exc})"
        )
    except Exception as exc:
        pytest.fail(f"your function raised {type(exc).__name__} on a valid graph: {exc}")
    if type(result) is not list:
        pytest.fail("return a list of node names, not a tuple, a generator or a set")
    return result


def test_examples_from_problem_md():
    assert _order(["m", "z", "a"], [("z", "a")]) == ["m", "z", "a"], (
        "m and z are both free at the start and m is smaller; a waits for z, so it comes last"
    )
    assert _order(["b", "a", "c"], [("b", "c")]) == ["a", "b", "c"], (
        "the ready names must be taken smallest first: a is free and smaller than b, so it "
        "goes first; a queue in arrival order gives b first"
    )


def test_empty_graph_returns_empty_list():
    assert _order([], []) == [], "an empty graph has the empty order"


def test_cycle_message_names_a_real_cycle():
    edges = [("x", "y"), ("y", "x")]
    with pytest.raises(ValueError) as caught:
        smallest_topological_order(["x", "y"], edges)
    message = str(caught.value)
    assert message.startswith("graph has a cycle: "), (
        "a cycle error must read 'graph has a cycle: x -> y -> x': the exact prefix, then the names"
    )
    names = [part.strip() for part in message[len("graph has a cycle: ") :].split("->")]
    assert len(names) >= 2 and names[0] == names[-1], (
        "the named cycle must start and end with the same name"
    )
    for step in pairwise(names):
        assert step in edges, "the named cycle uses a step that is not an edge of the graph"
