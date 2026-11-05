---
title: "Reverse-mode autograd engine"
minutes: 12
min_viable: "Read The system and Worked example, then implement p01 using Hint 1 if needed."
---

## The system

Reverse-mode differentiation turns a scalar computation into a graph, then
routes one scalar objective's sensitivity backward. Each `Value` node stores
a forward scalar `data`, a scalar `grad`, and its parent edges with local
partial derivatives. A node represents one result; an edge represents one
use of a result. These are different objects to count.

| Operation | Forward result | Local derivative to each parent |
|---|---|---|
| Addition | $a+b$ | $1,1$ |
| Multiplication | $ab$ | $b,a$ |
| Hyperbolic tangent | $t=\tanh(a)$ | $1-t^2$ |

Here $a$, $b$, and $t$ are real scalars. To differentiate a root scalar $L$,
first discover reachable nodes in parent-before-child order. Reset their
gradients to zero, seed $L$ with one, and traverse that order backward.
Every edge adds `child.grad * local_partial` into its parent. Our explicit
repeat rule resets reachable gradients each call; unrelated graphs retain
their existing gradients. First action: draw the graph of `x*x`.

## Worked example

Let $x=2$, $q=x\times x=4$, and $L=q\times q+q=20$.
The root addition sends one unit to the product $q\times q$ and one unit
directly to $q$. The product has two edges to $q$, each with partial $4$.
Consequently $q.grad=4+4+1=9$. Node $q$ also has two edges to $x$, each
with partial $2$, so $x.grad=9\times2+9\times2=36$.

Visit each node once to build the order, but retain duplicate edges.
Deduplicating the two occurrences of `x` would lose half the square's
derivative. Calling `L.backward()` again gives $36$ again under this
exercise's reset policy. Calling `q.backward()` afterward gives $4$ at $x$;
the differentiated root has changed.

## Why it works

The chain rule sums derivatives over every route from an intermediate value
to the objective. A reverse topological traversal ensures all child routes
have contributed before a node sends its accumulated signal to parents.
This is the dependency rule that makes graph reuse safe.

Reverse mode is efficient when there is one scalar objective and many
inputs: one backward traversal computes all reachable input derivatives.
A forward-mode method would instead propagate input directions. This
trade-off and the graph construction appear in [book: Goodfellow, Bengio
and Courville, Deep Learning, §6.5]. Local derivatives can be cached during
the forward pass because graph values stay fixed until backward finishes.

## Signal vs. noise

Focus on edge accumulation, traversal order, and gradient lifecycle. These
three ideas transfer directly to tensor frameworks. Operator overloading is
convenience, while the graph algorithm is the system. Our interface requires
both operands to be `Value` nodes; implicit conversion, division, broadcasting,
and gradient tracking switches can wait. The challenge here is dependency
bookkeeping, not deriving three elementary calculus rules.

## Where it breaks

Overwriting a parent's gradient loses shared paths. Recursing backward
immediately along each path can propagate before all contributions arrive.
Reusing old gradients without an explicit reset or accumulation policy makes
repeated calls surprising. This problem uses reset-on-backward, which differs
from frameworks that accumulate leaf gradients until the caller clears them.

Directly changing stored forward values would invalidate cached partials;
`data` is read-only. The graph is acyclic and limited to one hundred nodes,
so a recursive topological walk is sufficient. This is an educational scalar
engine, with no promise of deep-graph scalability. Inputs and intermediate
values must remain finite and within the stated scalar bound. Large tanh
inputs saturate; a near-zero derivative there is expected behavior.

## Interview angle

Explain graph construction, root seeding, reverse order, and additive
contributions in that sequence. Use `x*x` to expose duplicate-edge handling.
A likely follow-up asks about repeated backward calls; state the reset
policy before discussing implementation. Another asks why reverse mode suits
training; the loss is scalar and the parameter vector is large.

## Connections

Scalar chain-rule practice supplies the local derivatives. Dense-layer
backprop is the batched analogue, replacing scalar edges with matrix
vector-Jacobian products. A real autodiff library extends the same graph
principles to tensor shapes, device placement, saved activations, and memory
reclamation. The core conceptual source remains [book: Goodfellow, Bengio
and Courville, Deep Learning, §6.5].
