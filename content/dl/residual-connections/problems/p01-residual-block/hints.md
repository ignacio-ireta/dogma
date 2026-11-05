## Hint 1 — concept

The block has two parts that are added at the end: a shortcut that carries the input forward, and a branch that learns a correction. Decide which part the input feeds and which part the shape rule constrains.

## Hint 2 — formula / approach

Compute the branch first as `relu(x @ w1 + b1) @ w2 + b2`. Then pick the shortcut: `x` itself when `w_shortcut` is `None`, or `x @ w_shortcut` otherwise. Reject an identity shortcut whose width differs from `x`. The result is the shortcut plus the branch, with no activation after the sum.

## Hint 3 — code shape

Validate every shape and dtype before any arithmetic, so each `ValueError` names one rule. Then four lines: branch hidden, branch output, shortcut choice, and the sum. Never write into `x` and never call `detach()` on it.
