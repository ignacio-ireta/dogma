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
    """Return the number of edges on the longest path between two nodes of a binary tree.

    Iterative version: an explicit stack visits the nodes in post-order, so no recursion is used.
    """
    if root is not None and not all(hasattr(root, name) for name in ("val", "left", "right")):
        raise TypeError("root must be None or a tree node with val, left and right")
    if root is None:
        return 0

    heights: dict[int, int] = {}
    best = 0
    stack: list[tuple[TreeNode, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if not expanded:
            stack.append((node, True))
            if node.right is not None:
                stack.append((node.right, False))
            if node.left is not None:
                stack.append((node.left, False))
            continue
        left = heights[id(node.left)] if node.left is not None else -1
        right = heights[id(node.right)] if node.right is not None else -1
        best = max(best, left + right + 2)
        heights[id(node)] = max(left, right) + 1
    return best
