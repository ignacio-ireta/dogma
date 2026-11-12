# In-order traversal without recursion

Implement `inorder_traversal(root)`. It returns the node values of a binary tree in in-order
order: the whole left subtree, then the node, then the whole right subtree. Do it with an explicit
stack, not with recursion. A tree of 10,000 nodes can be a single chain, and a recursive walk needs one
stack frame per level, which exceeds Python's default recursion limit.

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
  accepted, because the tests build their own nodes. Children are `None` or nodes. The tree is acyclic.
- Returns: a Python `list` of the `val` attributes in in-order order. An empty tree returns `[]`.
- Values are returned exactly as stored. Negative numbers, zero and integers beyond 2**53 are
  returned unchanged.
- Must not mutate the tree: no node is changed, detached or reordered.
- Raises `TypeError` when `root` is neither `None` nor an object with `val`, `left` and `right`.
- Time O(n) and extra space O(h), where h is the height of the tree. A height of 10,000 must be handled.

## Examples

```python
#        2
#       / \
#      1   3
inorder_traversal(TreeNode(2, TreeNode(1), TreeNode(3)))
# [1, 2, 3]

#   1
#    \
#     2
#      \
#       3
inorder_traversal(TreeNode(1, None, TreeNode(2, None, TreeNode(3))))
# [1, 2, 3]

inorder_traversal(None)
# []
```

```python
# A single node, and a left-leaning chain of three nodes:
inorder_traversal(TreeNode(7))  # [7]
inorder_traversal(TreeNode(3, TreeNode(2, TreeNode(1))))  # [1, 2, 3]
```

A right-leaning chain of 10,000 nodes with values 0 to 9999 returns `[0, 1, ..., 9999]`. A recursive
walk fails on it with `RecursionError`.

## Constraints

- Allowed: built-in types and the standard library, such as `list` used as a stack.
- Not allowed: `sys` and `threading` (both would let you raise the recursion limit), and recursion
  of any depth that depends on the tree's height. Any call to `sys.setrecursionlimit` is also
  forbidden.
  Neither module is needed for this problem: a `list` stack and plain loops are enough.

## Done looks like

Your function returns the in-order sequence on the examples, on the empty tree, and on a 10,000-deep
chain. The hidden tests compare with a reference walk on seeded random trees, check the left-leaning
and right-leaning chains of depth 10,000, check a balanced tree, and confirm that the values are kept
exactly, that the return type is `list`, and that the tree is unchanged after the call. A pre-order or
post-order answer, a walk that drops right subtrees, and a recursive walk that fails on the deep chain
each fail with the mistake named.

The hidden tests draw their bulk inputs from the real dataset `wine` (see `docs/DATASETS.md`).
