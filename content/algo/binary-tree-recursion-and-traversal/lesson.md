---
title: "Binary tree recursion and traversal"
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01 with a list used as the stack."
---

## The system

A binary tree node $v$ has a value $v.\text{val}$ and two children $v.\text{left}$ and
$v.\text{right}$, each a node or `None`. Let $T(v)$ be the subtree rooted at $v$. An empty
subtree is `None`.

Every recursive tree function has the same shape:

1. **Base case.** If `node is None`, return the answer for the empty tree.
2. **Recurse.** Get the answers for `node.left` and `node.right`.
3. **Combine.** Build this node's answer from the node's value and the two child answers.

The three depth-first traversals differ only in step 3, where the visit happens:

| Order | Visit happens | Output for the tree in the worked example |
|---|---|---|
| Pre-order | before both children | 4, 2, 1, 3, 6, 7 |
| In-order | between the children | 1, 2, 3, 4, 6, 7 |
| Post-order | after both children | 1, 3, 2, 7, 6, 4 |
| Level-order | by depth, using a queue | 4 \| 2, 6 \| 1, 3, 7 |

In-order on a binary search tree returns the values in sorted order. That is why it is the usual
choice when a question mentions sorted output.

**Two returns.** Some quantities need two things from a child. Height needs the depth of the
subtree. A diameter needs the depth and also the best path seen so far. Return the depth and update
the best value in a non-local variable. The conventions to fix up front: the height of `None` is
$-1$ in edges, so a leaf has height $0$; and a path through node $v$ uses $h(\text{left}) + h(\text{right}) + 2$
edges, with the $-1$ convention covering a missing child.

## Worked example

The tree has five edges to walk:

```text
        4
      /   \
     2     6
    / \     \
   1   3     7
```

Pre-order visits 4 first, then descends left. In-order must finish the left subtree before it
emits 4, so it emits 1, 2, 3, then 4, then the right subtree 6, 7.

The in-order walk with an explicit stack, step by step:

| Step | Action | Stack (top right) | Output |
|---|---|---|---|
| 1 | push 4, 2, 1 while going left | 4, 2, 1 | |
| 2 | pop 1, emit it, go right (none) | 4, 2 | 1 |
| 3 | pop 2, emit it, go right to 3, push 3 | 4, 3 | 1, 2 |
| 4 | pop 3, emit it, go right (none) | 4 | 1, 2, 3 |
| 5 | pop 4, emit it, go right to 6, push 6 | 6 | 1, 2, 3, 4 |
| 6 | pop 6, emit it, go right to 7, push 7 | 7 | 1, 2, 3, 4, 6 |
| 7 | pop 7, emit it, stop | | 1, 2, 3, 4, 6, 7 |

The diameter is 4 edges: the path 1, 2, 4, 6, 7, or 3, 2, 4, 6, 7. The height of the root is 2 edges.

## Why it works

Each call handles one subtree, and the recursion gives every subtree the same work. The three steps cover every node exactly once. The iterative version keeps the same invariant on
an explicit list: the stack holds the nodes whose left subtree is done but whose visit has not
happened.

The cost is linear in the number of nodes: each node is pushed and popped once.

## Signal vs. noise

- **Signal:** the base case on `None`, the visit position, and the two-return pattern.
- **Signal:** the recursion depth. A recursive function uses one stack frame per level. Python's
  default limit is about 1000, so a chain of 10,000 nodes fails with `RecursionError` in a
  recursive traversal. The explicit list has no such limit.
- **Noise for now:** threaded trees, Morris traversal and the rotation rules of balanced trees.

## Where it breaks

- **Missing base case.** Dereferencing `None.left` raises `AttributeError`. Check `None` first.
- **Wrong empty value.** A height of 0 for `None` makes a leaf look like height 1, and every path
  length shifts by one. Use $-1$ for the height of `None` in edges, or count nodes and say so.
- **A diameter that stops at the root.** The longest path can be inside one subtree. Update the best
  value at every node, not only the root.
- **Deep chains.** A right-leaning chain of 10,000 nodes is a valid tree. Recursion fails on it, so
  use the stack version.
- **Mutating while walking.** Clearing child pointers changes the tree for the next caller.

## Interview angle

Say the base case and the visit position before you write the code. For a two-return problem, say
what each call returns and what the global best means. Mention the depth limit in one sentence:
recursion is clean, but a skewed tree can be deep, so the explicit stack is the safe version.

## Connections

`algo/stacks-and-queues` gives the stack and queue that the iterative walks use. Level order is a
queue; the other three orders are stacks in disguise. The post-order pattern also underlies tree
dynamic programming, and `algo/bfs-and-dfs-on-graphs-and-grids` generalises the walks to graphs.
