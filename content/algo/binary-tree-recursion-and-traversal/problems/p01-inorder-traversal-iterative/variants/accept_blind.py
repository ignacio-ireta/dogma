from __future__ import annotations


class TreeNode:
    """A binary tree node. `left` and `right` are TreeNode objects or None."""

    def __init__(
        self, val: int, left: TreeNode | None = None, right: TreeNode | None = None
    ) -> None:
        self.val = val
        self.left = left
        self.right = right


def inorder_traversal(root: TreeNode | None) -> list[int]:
    """Return the values of the tree in in-order order, using an explicit stack."""
    if root is not None and not all(hasattr(root, name) for name in ("val", "left", "right")):
        raise TypeError("root must be None or a tree node with val, left and right")

    result: list[int] = []
    stack: list = []
    node = root
    while stack or node is not None:
        while node is not None:
            stack.append(node)
            node = node.left
        node = stack.pop()
        result.append(node.val)
        node = node.right
    return result
