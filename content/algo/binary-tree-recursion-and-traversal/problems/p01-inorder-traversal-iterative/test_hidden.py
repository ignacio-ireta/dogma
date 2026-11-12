"""Hidden tests: a recursive reference on seeded random trees, deep and degenerate shapes, exact
values, return type, and the tree left unchanged.

The reference is a small recursive walk defined here, and it is used only on trees of depth below
60. Learner code is imported from submission.py; nothing here imports solution.py.
"""

from __future__ import annotations

import functools
import random
from typing import Any

import pytest
from submission import inorder_traversal

from dogma_data import load


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


def _reference(node: Any) -> list[int]:
    """Independent recursive in-order walk, used only on shallow trees."""
    if node is None:
        return []
    return [*_reference(node.left), node.val, *_reference(node.right)]


@functools.cache
def _real_values() -> tuple[int, ...]:
    """Real whole numbers of both signs: the proline level of 178 wines, with 750 subtracted."""
    return tuple(int(v) - 750 for v in load("wine").X[:, 12])


def _random_tree(rng: random.Random, size: int) -> Any:
    """A binary tree with `size` nodes: each new node hangs from a random free slot.

    The shape is drawn, because the shape is what the problem is about. The values in the nodes
    are real measurements (wine proline levels, centred), which repeat and have both signs.
    """
    values = _real_values()
    nodes = [_Node(rng.choice(values))]
    slots = [(nodes[0], "left"), (nodes[0], "right")]
    for _ in range(size - 1):
        node = _Node(rng.choice(values))
        parent, side = slots.pop(rng.randrange(len(slots)))
        setattr(parent, side, node)
        slots.extend([(node, "left"), (node, "right")])
        nodes.append(node)
    return nodes[0]


def _snapshot(root: Any) -> list[tuple[int, int, int, int]]:
    """Identity, value and child identities of every node, in pre-order, without recursion."""
    out = []
    stack = [root] if root is not None else []
    while stack:
        node = stack.pop()
        out.append(
            (
                id(node),
                node.val,
                id(node.left) if node.left is not None else 0,
                id(node.right) if node.right is not None else 0,
            )
        )
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    return out


def _expect(got: Any, expected: Any, mistake: str) -> None:
    if got != expected:
        pytest.fail(mistake)


def test_matches_recursive_reference_on_seeded_random_trees():
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic trees, not security.
    for _ in range(200):
        size = rng.randrange(0, 60)
        root = _random_tree(rng, size) if size else None
        expected = _reference(root)  # computed before the call: the learner must not change it
        _expect(
            _call(root),
            expected,
            "in-order output differs from the recursive reference: check that the left subtree "
            "comes before the node and the right subtree comes after it",
        )


def test_left_leaning_chain_of_ten_thousand_nodes_comes_back_bottom_up():
    root = None
    for value in range(10000):
        root = _Node(value, root, None)
    _expect(
        _call(root),
        list(range(10000)),
        "a left-leaning chain must come back bottom-up: the deepest left node (0) is visited first",
    )


def test_balanced_tree_of_depth_nine_returns_sorted_values():
    def build(lo: int, hi: int) -> Any:
        if lo > hi:
            return None
        mid = (lo + hi) // 2
        return _Node(mid, build(lo, mid - 1), build(mid + 1, hi))

    root = build(0, 1022)  # a perfect tree of 1023 nodes, depth 9
    _expect(
        _call(root),
        list(range(1023)),
        "a balanced search tree must give its values in sorted order; check the visit position",
    )


def test_subtrees_on_both_sides_keep_their_children():
    #         5
    #      /     \
    #     3       8
    #    / \     /
    #   1   4   7
    root = _Node(5, _Node(3, _Node(1), _Node(4)), _Node(8, _Node(7), None))
    _expect(
        _call(root),
        [1, 3, 4, 5, 7, 8],
        "the walk drops a subtree: the right child of a node must be visited after the node",
    )


def test_negative_zero_and_huge_values_are_returned_exactly():
    values = [-(10**18), 0, 2**53 + 1, -1, 10**30]
    root = None
    for value in sorted(values, reverse=True):
        root = _Node(value, None, root)
    _expect(
        _call(root),
        sorted(values),
        "values must come back exactly as stored: no rounding, no sign change, no truncation",
    )


def test_result_is_a_plain_list_of_the_values():
    root = _Node(2, _Node(1), _Node(3))
    result = _call(root)
    _expect(
        type(result),
        list,
        "return a list of values, not a tuple, a generator or the nodes themselves",
    )
    _expect(
        all(type(item) is int for item in result),
        True,
        "the list must hold the node values, not the node objects",
    )


def test_tree_is_unchanged_after_the_walk():
    rng = random.Random(7)  # noqa: S311 -- seeded synthetic tree.
    root = _random_tree(rng, 40)
    before = _snapshot(root)
    _call(root)
    _expect(
        _snapshot(root),
        before,
        "the walk changed the tree: do not detach, reset or reorder any child while walking",
    )


def test_wrong_root_type_raises_type_error():
    failure = None
    try:
        inorder_traversal(5)  # type: ignore[arg-type]  # invalid root
    except TypeError:
        return
    except RecursionError:
        failure = "a non-tree root must raise TypeError, not RecursionError"
    except Exception as exc:
        failure = f"a non-tree root must raise TypeError, instead it raised {type(exc).__name__}"
    if failure is not None:
        pytest.fail(failure)
    pytest.fail("a non-tree root was accepted: check the root before walking it")


class _ValOnly:
    def __init__(self) -> None:
        self.val = 1


class _ValAndLeftOnly:
    def __init__(self) -> None:
        self.val = 1
        self.left = None


def test_object_missing_one_tree_attribute_raises_type_error() -> None:
    for bad in (_ValOnly(), _ValAndLeftOnly()):
        try:
            inorder_traversal(bad)  # type: ignore[arg-type]
        except TypeError:
            continue
        except Exception as exc:
            pytest.fail(
                "a root missing one of val, left or right must raise TypeError, instead it raised "
                f"{type(exc).__name__}"
            )
        pytest.fail(
            "a root missing one of val, left or right was accepted: check all three attributes "
            "before walking the tree"
        )
