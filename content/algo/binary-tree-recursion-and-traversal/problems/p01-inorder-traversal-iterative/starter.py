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
    """Return the values of the tree in in-order order, using an explicit stack.

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

    Notes
    -----
    Use a list as the stack, not recursion. A chain of 10,000 nodes is valid input and is deeper
    than Python's default recursion limit.
    """
    raise NotImplementedError
