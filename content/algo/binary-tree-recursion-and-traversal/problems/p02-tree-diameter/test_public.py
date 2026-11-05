"""Public tests: the examples from problem.md, the empty tree and a single node.

The learner's code is imported from submission.py. The tests build their own nodes.
"""

from __future__ import annotations

from typing import Any

import pytest
from submission import diameter


class _Node:
    """Test-side tree node with the three attributes the contract requires."""

    def __init__(self, val: int, left: Any = None, right: Any = None) -> None:
        self.val = val
        self.left = left
        self.right = right


def _call(root: Any) -> Any:
    failure = None
    try:
        result = diameter(root)
    except RecursionError:
        failure = (
            "RecursionError: the tree is deeper than the recursion limit allows for a recursive "
            "helper; the public examples are shallow, so check the base case for None"
        )
    except Exception as exc:
        failure = f"your function raised {type(exc).__name__} on a valid tree: {exc}"
    if failure is not None:
        pytest.fail(failure)
    if type(result) is not int:
        pytest.fail("return a plain int edge count, not a bool, a float or a list of nodes")
    return result


def test_empty_tree_and_single_node_have_diameter_zero():
    assert _call(None) == 0, "the empty tree has diameter 0"
    assert _call(_Node(5)) == 0, (
        "a single node has diameter 0: a path from a node to itself has no edges"
    )


def test_two_leaves_under_one_root_are_two_edges_apart():
    assert _call(_Node(1, _Node(2), _Node(3))) == 2, (
        "two leaves under one root are 2 edges apart through the root"
    )


def test_chain_of_three_nodes_has_two_edges():
    assert _call(_Node(1, _Node(2, _Node(3)))) == 2, (
        "a chain of three nodes has two edges between its ends"
    )


def test_longest_path_can_lie_inside_one_subtree():
    # Root t has a leaf on the left and a subtree r on the right. Inside r, the path from the end of
    # r's left chain to the end of its right chain has 6 edges. Through the root the best path is
    # 5 edges, so a diameter that is only checked at the root gives 5 instead of 6.
    leaf = _Node(1)
    left_chain = _Node(3, _Node(4, _Node(5)))
    right_chain = _Node(7, None, _Node(8, None, _Node(9)))
    r = _Node(2, left_chain, right_chain)
    root = _Node(0, leaf, r)
    assert _call(root) == 6, (
        "the longest path lies inside the right subtree, with 6 edges; a diameter taken only at "
        "the root gives 5"
    )
