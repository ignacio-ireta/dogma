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
    """Wrong on purpose: return the values of the tree in in-order order, using an explicit stack.

    Parameters
    ----------
    root : TreeNode or None
        The root of a binary tree. Any object with val, left and right attributes is accepted.
        Must not be mutated.

    Returns
    -------
    list[int]
        Values in in-order order: left subtree, node, right subtree. Empty for None.

    Raises
    ------
    TypeError
        When root is neither None nor a tree node.
    """
    if root is not None and not (
        hasattr(root, "val") and hasattr(root, "left") and hasattr(root, "right")
    ):
        raise TypeError("root must be None or a tree node with val, left and right")

    return _walk(root)


def _walk(node: TreeNode | None) -> list[int]:
    if node is None:
        return []
    return _walk(node.left) + [node.val] + _walk(node.right)  # noqa: RUF005
