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

    Parameters
    ----------
    root : TreeNode or None
        The root of a binary tree. Any object with val, left and right attributes is accepted.
        Must not be mutated. Values are ignored.

    Returns
    -------
    int
        The diameter in edges. 0 for the empty tree and for a single node.

    Raises
    ------
    TypeError
        When root is neither None nor a tree node.

    Notes
    -----
    Recursion is allowed for trees of height up to 900. The path may pass through the root or lie
    inside one subtree, so the best value must be updated at every node.
    """
    raise NotImplementedError
