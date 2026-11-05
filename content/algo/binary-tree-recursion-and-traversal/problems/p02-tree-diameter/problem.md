# Diameter of a binary tree

The diameter of a binary tree is the largest number of edges on any path between two nodes. The
path does not have to pass through the root: the longest path can lie entirely inside one subtree.
Implement `diameter(root)`, which returns that number.

The starter defines the node class. Use its attributes, and do not change the class.

```python
class TreeNode:
    def __init__(
        self, val: int, left: "TreeNode | None" = None, right: "TreeNode | None" = None
    ) -> None:
        self.val = val
        self.left = left
        self.right = right
```

## Contract

- `root`: a `TreeNode` or `None`. Any object with the attributes `val`, `left` and `right` is
  accepted. The tree is acyclic and its values are ignored.
- Returns: a Python `int`, the number of edges on the longest path. A tree with one node has diameter
  0. The empty tree has diameter 0.
- Raises `TypeError` when `root` is neither `None` nor an object with `val`, `left` and `right`.
- Must not mutate the tree.
- Time O(n) for n nodes; the extra space may be O(h) for a height h.
- Recursion is allowed. The hidden trees have height at most 900, which fits in Python's default
  recursion limit, so one recursive call per level is fine.

## Examples

```python
diameter(None)
# 0                      empty tree

diameter(TreeNode(5))
# 0                      one node: a path from a node to itself has no edges

diameter(TreeNode(1, TreeNode(2), TreeNode(3)))
# 2                      2 - 1 - 3
```

```python
#     1
#    /
#   2
#  /
# 3
diameter(TreeNode(1, TreeNode(2, TreeNode(3))))
# 2                      3 - 2 - 1 is a path of two edges

#       1
#      / \
#     2   3
#    /     \
#   4       5
#  /
# 6
diameter(TreeNode(1, TreeNode(2, TreeNode(4, TreeNode(6))), TreeNode(3, None, TreeNode(5))))
# 5                      6 - 4 - 2 - 1 - 3 - 5 has five edges
```

## Constraints

- Allowed: built-in types and the standard library.
- Not allowed: `sys` and `threading`, and any call to `sys.setrecursionlimit`. Recursion within the
  depth limit is allowed; raising the limit is not.
  Neither module is needed for this problem.
- Not allowed: rebuilding the tree as a graph library object. A plain traversal is enough.

## Done looks like

Your function matches the examples, the empty tree and the one-node tree. The hidden tests compare with a
breadth-first search from every node on seeded random trees, check a chain of 900 nodes, a perfect tree,
and trees whose longest path lies inside one subtree and not through the root. They also check that the
result is an `int`, that values do not affect the answer, and that the tree is left unchanged. A
diameter that stops at the root, a height that is used as the diameter, and a height of 0 for the empty
tree each fail with the mistake named.
