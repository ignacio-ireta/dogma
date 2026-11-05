"""Public tests: the examples from problem.md, a single node, and a deep chain.

The learner's code is imported from submission.py. The tests build their own nodes, so the
learner's class is never needed.
"""

from __future__ import annotations

from typing import Any

import pytest
from submission import inorder_traversal


class _Node:
    """Test-side tree node with the three attributes the contract requires."""

    def __init__(self, val: int, left: Any = None, right: Any = None) -> None:
        self.val = val
        self.left = left
        self.right = right


def _call(root: Any) -> Any:
    failure = None
    try:
        result = inorder_traversal(root)
    except RecursionError:
        failure = (
            "RecursionError on a deep tree: a recursive walk keeps one stack frame per level, and "
            "a chain deeper than the default limit (about 1000) exhausts them. Keep the pending "
            "nodes in a list, as the explicit stack in the lesson does"
        )
    except Exception as exc:
        failure = f"your function raised {type(exc).__name__} on a valid tree: {exc}"
    if failure is not None:
        pytest.fail(failure)
    if type(result) is not list:
        pytest.fail("return a list of values, not a tuple, a generator or the nodes themselves")
    return result


def test_examples_from_problem_md():
    assert _call(_Node(2, _Node(1), _Node(3))) == [1, 2, 3], (
        "the balanced three-node tree must give left, node, right: 1, 2, 3"
    )
    assert _call(_Node(1, None, _Node(2, None, _Node(3)))) == [1, 2, 3], (
        "a right-leaning chain is already in order: visit each node before its right child"
    )
    assert _call(None) == [], "the empty tree must return an empty list"


def test_single_node_returns_its_value():
    assert _call(_Node(-7)) == [-7], "a single node is its own in-order walk"


def test_ten_thousand_deep_chain_is_walked_without_recursion():
    root = None
    for value in range(9999, -1, -1):
        root = _Node(value, None, root)
    assert _call(root) == list(range(10000)), (
        "the 10,000-deep right chain must come back as 0..9999; a recursive walk raises "
        "RecursionError here, so keep the stack in a list"
    )
