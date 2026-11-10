from __future__ import annotations


class TreeNode:
    """A binary tree node. `left` and `right` are TreeNode objects or None."""

    def __init__(
        self, val: int, left: TreeNode | None = None, right: TreeNode | None = None
    ) -> None:
        self.val = val
        self.left = left
        self.right = right


def diameter(root: TreeNode | None) -> int:
    """Return the number of edges on the longest path between two nodes of a binary tree."""
    if root is not None and not all(hasattr(root, name) for name in ("val", "left", "right")):
        raise TypeError("root must be None or a tree node with val, left and right")

    best = 0

    def depth(node) -> int:
        # Number of nodes on the longest downward path from node; 0 for None.
        nonlocal best
        if node is None:
            return 0
        left_depth = depth(node.left)
        right_depth = depth(node.right)
        best = max(best, left_depth + right_depth)
        return 1 + max(left_depth, right_depth)

    depth(root)
    return int(best)
