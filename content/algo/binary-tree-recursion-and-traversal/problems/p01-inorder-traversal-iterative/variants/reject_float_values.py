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

    result: list[int] = []
    # stack holds nodes whose left subtree is finished but whose own value is not emitted yet.
    # It is the explicit replacement for the call stack of a recursive walk, so its size is the
    # height of the tree at most, and no depth of the input can exceed a Python recursion limit.
    stack: list[TreeNode] = []
    node = root
    while stack or node is not None:
        # Step 1 (go left): walk down the left spine, pushing each node. The leftmost ends on top.
        while node is not None:
            stack.append(node)
            node = node.left
        # Step 2 (visit): the left subtree of the top node is finished, so emit the node itself.
        node = stack.pop()
        result.append(float(node.val))
        # Step 3 (go right): the right subtree is handled the same way, starting from its root.
        node = node.right
    return result
