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
    """Wrong on purpose: return the number of edges on the longest path between two nodes of a binary tree.

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
    """  # noqa: E501
    if root is not None and not all(hasattr(root, name) for name in ("val", "left", "right")):
        raise TypeError("root must be None or a tree node with val, left and right")

    # best holds the longest path found so far, in edges. Every node can be the top of a path, not
    # only the root, so best is updated inside the recursion and returned at the end.
    best = 0

    def plain_height(node: TreeNode | None) -> int:
        if node is None:
            return -1
        return max(plain_height(node.left), plain_height(node.right)) + 1

    def height(node: TreeNode | None) -> int:
        """Return the height of node in edges, and update best with the path through node.

        The height of None is -1, so that a leaf has height 0. A path through node goes down its
        left subtree to a node at depth left + 1 edges, then down its right subtree, which is
        right + 1 edges. Together that is left + right + 2 edges, and the -1 convention makes a
        missing child contribute the right amount.
        """
        nonlocal best
        if node is None:
            return -1
        left = height(node.left)
        right = height(node.right)
        best = max(best, plain_height(node.left) + plain_height(node.right) + 2)
        return max(left, right) + 1

    height(root)
    return best
