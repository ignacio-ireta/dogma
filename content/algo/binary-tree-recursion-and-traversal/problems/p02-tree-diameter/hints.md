## Hint 1 — concept

The longest path in a tree is the sum of two downward legs that meet at one node: the top of the path.
That node need not be the root. So ask, for every node, how deep its left subtree goes and how deep its
right subtree goes, and what path that combination makes.

## Hint 2 — formula / approach

Define the height of a node as the number of edges on the longest downward path from it. A path with
top node v has length $h(\text{left}) + h(\text{right}) + 2$, where the two extra edges reach the children.
Use $-1$ as the height of `None`, so a missing child gives the correct length with no special case.
Return $\max(h(\text{left}), h(\text{right})) + 1$ to the parent, and keep the largest path length seen at
any node in a variable that outlives the recursion.

## Hint 3 — code shape

Write one recursive helper that returns a height and updates a `best` variable declared `nonlocal`. The
helper returns $-1$ for `None`. Its body has three lines: recurse on both children, update the best value
with the sum, and return the larger child height plus one. Wrap the helper in the public function, which
returns `best` after calling the helper on the root.
