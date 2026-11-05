## Hint 1 — concept

Separate graph discovery from gradient propagation. Shared nodes need one visit but may have several contributing edges.

## Hint 2 — formula / approach

Build parent-before-child order using a visited set. Reverse it after clearing reachable gradients and seeding the root. Every edge adds the child gradient times its local partial.

## Hint 3 — code shape

Store parent/partial pairs on each result. Let a small recursive visitor append a node after its parents. Keep duplicate parent edges even when the visitor uses a set.
