"""Hidden tests: an all-pairs breadth-first oracle on seeded random trees, long chains, a perfect
tree, values that must not matter, the return type, non-mutation and the invalid-root rule.

The oracle is a breadth-first search from every node of the tree viewed as an undirected graph. It
shares no logic with the recursive height formula. Learner code is imported from submission.py.
"""

from __future__ import annotations

import functools
import random
from collections import deque
from typing import Any

import pytest
from submission import diameter

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
        result = diameter(root)
    except RecursionError:
        failure = (
            "RecursionError on a tree of height up to 900: the recursion must stop at None and "
            "must not recurse on the same node twice"
        )
    except Exception as exc:
        failure = f"your function raised {type(exc).__name__} on a valid tree: {exc}"
    if failure is not None:
        pytest.fail(failure)
    if type(result) is not int:
        pytest.fail("return a plain int edge count, not a bool, a float or a list of nodes")
    return result


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


def _oracle(root: Any) -> int:
    """Largest breadth-first distance between any two nodes of the tree, viewed as a graph."""
    if root is None:
        return 0
    ids: dict[int, int] = {}
    adjacency: dict[int, list[int]] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        here = ids.setdefault(id(node), len(ids))
        adjacency.setdefault(here, [])
        for child in (node.left, node.right):
            if child is not None:
                there = ids.setdefault(id(child), len(ids))
                adjacency.setdefault(there, []).append(here)
                adjacency[here].append(there)
                stack.append(child)
    best = 0
    for start in adjacency:
        dist = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for nxt in adjacency[current]:
                if nxt not in dist:
                    dist[nxt] = dist[current] + 1
                    queue.append(nxt)
        best = max([best, *dist.values()])
    return best


def _snapshot(root: Any) -> list[tuple[int, int, int, int]]:
    """Identity, value and child identities of every node, collected without recursion."""
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


def test_matches_all_pairs_breadth_first_oracle_on_seeded_random_trees():
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic trees, not security.
    for _ in range(300):
        size = rng.randrange(0, 40)
        root = _random_tree(rng, size) if size else None
        expected = _oracle(root)  # computed before the call: the learner must not change it
        _expect(
            _call(root),
            expected,
            "diameter differs from the all-pairs oracle: the longest path can lie inside one "
            "subtree, so update the best value at every node, and count edges, not nodes",
        )


def test_chain_of_900_nodes_has_899_edges():
    root = None
    for value in range(899, -1, -1):
        root = _Node(value, None, root)
    _expect(
        _call(root),
        899,
        "a chain of 900 nodes has 899 edges between its ends: count edges, not nodes",
    )


def test_perfect_tree_of_depth_nine_has_diameter_eighteen():
    def build(depth: int) -> Any:
        if depth == 0:
            return None
        return _Node(depth, build(depth - 1), build(depth - 1))

    root = build(10)  # 1023 nodes; every leaf is 9 edges below the root
    _expect(
        _call(root),
        18,
        "a perfect tree of depth 9 has a diameter of 18 edges: two legs of 9 through the root",
    )


def test_longest_path_inside_one_subtree_is_found():
    # Root with a leaf on the left; the right subtree has two chains of length 3 meeting at its top.
    right = _Node(
        2,
        _Node(3, _Node(4, _Node(5))),
        _Node(7, None, _Node(8, None, _Node(9))),
    )
    root = _Node(0, _Node(1), right)
    _expect(
        _call(root),
        6,
        "the longest path lies inside the right subtree; a diameter taken only at the root gives 5",
    )


def test_values_do_not_change_the_diameter():
    shape = [(0, 1), (0, 2), (1, 3), (1, 4), (4, 5)]

    def build(values: list[int]) -> Any:
        nodes = [_Node(v) for v in values]
        for parent, child in shape:
            side = "left" if nodes[parent].left is None else "right"
            setattr(nodes[parent], side, nodes[child])
        return nodes[0]

    small = build([0, 1, 2, 3, 4, 5])
    huge = build([-(10**18), 2**53 + 1, 0, -1, 10**30, 7])
    _expect(
        _call(huge),
        _call(small),
        "the diameter depends only on the shape: values must not change the answer",
    )


def test_result_is_a_plain_int():
    root = _Node(1, _Node(2), _Node(3))
    result = _call(root)
    _expect(
        type(result),
        int,
        "return a plain int edge count, not a bool, a float or a list of nodes",
    )


def test_tree_is_unchanged_after_the_call():
    rng = random.Random(11)  # noqa: S311 -- seeded synthetic tree.
    root = _random_tree(rng, 35)
    before = _snapshot(root)
    _call(root)
    _expect(
        _snapshot(root),
        before,
        "the call changed the tree: do not detach or reset children while measuring heights",
    )


def test_wrong_root_type_raises_type_error():
    failure = None
    try:
        diameter(5)  # type: ignore[arg-type]  # invalid root
    except TypeError:
        return
    except RecursionError:
        failure = "a non-tree root must raise TypeError, not RecursionError"
    except Exception as exc:
        failure = f"a non-tree root must raise TypeError, instead it raised {type(exc).__name__}"
    if failure is not None:
        pytest.fail(failure)
    pytest.fail("a non-tree root was accepted: check the root before measuring it")


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
            diameter(bad)  # type: ignore[arg-type]
        except TypeError:
            continue
        except Exception as exc:
            pytest.fail(
                "a root missing one of val, left or right must raise TypeError, instead it raised "
                f"{type(exc).__name__}"
            )
        pytest.fail(
            "a root missing one of val, left or right was accepted: check all three attributes "
            "before measuring the tree"
        )


def test_child_reads_grow_linearly_with_the_number_of_nodes():
    # Contract: time O(n). Count the reads of the child attributes on a chain of 300 nodes.
    # A linear walk reads each child a bounded number of times; 50 reads per node is generous.
    # A walk that rescans subtrees reads on the order of n squared times.
    reads = [0]

    class _Counted(_Node):
        def __getattribute__(self, name):
            if name in ("left", "right"):
                reads[0] += 1
            return object.__getattribute__(self, name)

    root = None
    for value in range(299, -1, -1):
        root = _Counted(value, None, root)
    assert _call(root) == 299, "the diameter of a chain is the number of edges between its two ends"
    assert reads[0] <= 50 * 300, (
        "the diameter must be found in one linear pass: child attributes were read too often"
    )
