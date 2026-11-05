# Interview notes — Binary tree recursion and traversal

## Q1. Traverse a binary tree in-order without recursion.
**60-second answer.** I keep an explicit stack and a current pointer. The inner loop walks left
and pushes every node on the way. When it cannot go left, it pops a node, emits its value, and
moves to the right child. The loop ends when the stack is empty and the pointer is `None`. Each
node is pushed once and popped once, so the walk is O(n) time and O(h) extra space, where h is the
height. The reason for the stack is that a skewed tree can be as deep as n.
**Follow-ups.** Why not recurse? It is clean, but a 10,000-deep chain exceeds the default recursion
limit. Can you do pre-order with a stack? Yes: pop a node, emit it, push right, then left. Why does
the order of the pushes matter? The stack is last in, first out, so the left child must go on last.

## Q2. Find the longest path between any two nodes of a binary tree.
**60-second answer.** A path either passes through the root or lies inside one subtree. So at
every node I combine the heights of the two children, add two edges for the links to them, and keep
the largest total in a variable outside the recursion. The function returns the height of the
subtree, which the parent needs. The height of `None` is $-1$, so a leaf has height 0 and the formula
needs no special case for a missing child.
**Follow-ups.** Why not take the diameter of the root only? The longest path can sit in one subtree.
What is the cost? One post-order pass, O(n). Can it be done without recursion? Yes, with a post-order
stack that tracks the height of each finished child.

## Q3. How do you check a tree function quickly?
**60-second answer.** Compare it with a second, slow method on small random trees: for in-order, a
recursive walk; for diameter, a breadth-first search from every node. Then test the shapes that
break things: empty, single node, a long chain and a perfect tree. The chain catches recursion
depth and the perfect tree catches a wrong combine step.
**Follow-ups.** Why not use the function on the input only? A function that returns plausible output
on the examples can still miss a subtree. An independent oracle catches that.

## Common mistakes
- Missing the `None` check, so `node.left` raises `AttributeError`.
- Using 0 as the height of `None`, which shifts every path length by one.
- Updating the diameter only at the root.
- Recursing on a skewed tree of depth close to 1000 and then blaming the input.
- Mixing up the visit position of pre-, in- and post-order.
