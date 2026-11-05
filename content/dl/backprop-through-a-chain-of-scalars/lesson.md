---
title: Backprop through a chain of scalars
minutes: 12
min_viable: Read The system and Worked example, then implement p01.
---

## The system

Reverse-mode differentiation turns a scalar loss into derivatives for all its inputs. First evaluate a computational graph forward and retain the intermediate values needed by local derivatives. Then move backward from the loss. For each edge, multiply the received gradient by the local derivative; add contributions when several edges lead to the same value.

For input scalars $x,w,b,t\in\mathbb{R}$, p01 fixes this graph:

```text
p = w*x -> z = p+b -> h = tanh(z)
                         |       |
                       q = h*w  e = exp(h)
                         \       /
                          y = q+e -> r = y-t -> L = r*r
```

Here $L$ is scalar loss and $t$ is a target. Weight $w$ is reused at $p$ and $q$; activation $h$ is reused at $q$ and $e$. First action: circle the reused values before calculating any derivative.

## Worked example

Choose $x=w=b=t=0$. The forward pass gives $p=z=h=q=0$, $e=y=r=1$, and $L=1$. Start backward with $\partial L/\partial L=1$. Since $L=r^2$, the gradient into $r$ is $2r=2$. Subtraction gives gradients $2$ into $y$ and $-2$ into $t$.

Addition sends $2$ into both $q$ and $e$. Node $q=hw$ contributes $2w=0$ into $h$ and $2h=0$ into $w$. Node $e=\exp(h)$ contributes $2e=2$ into $h$. Their sum is two. Tanh's local derivative is one here, so $z$, $p$ and $b$ receive two. Multiplication at $p$ contributes zero to $x$ and $w$. The final gradients are $[0,0,2,-2]$ in input order.

## Why it works

Let $g_v=\partial L/\partial v$ denote the adjoint of scalar node $v$. If $v=f(u)$, the chain rule gives $g_u\mathrel{+}=g_v f'(u)$. The plus-equals matters: different uses are different paths through the same loss. Overwriting a contribution throws away a valid path.

The [local backpropagation note](../../../sources/library/AI-ML/10-Backpropagation/Rumelhart%20Hinton%20Williams%20Backpropagation%201986.md) explains the historical hidden-layer credit-assignment problem. This exercise exposes the same mechanism with scalar nodes, so no tensor shapes or automatic graph traversal hide the chain rule.

## Signal vs. noise

The signal is local derivative, reverse traversal and accumulation. Store $h$ and $e$ during the forward pass because tanh's derivative uses $1-h^2$ and exp's derivative uses $e$. A correct forward loss alone cannot establish that every backward path is included.

The dictionary keys are bookkeeping: return exactly the four input names and plain float gradients. No optimizer update happens here. A gradient describes local sensitivity; choosing a learning rate and changing weights is a later operation.

## Where it breaks

Saturated tanh can have a derivative that rounds to zero. That does not erase the direct $w$ contribution through $q=hw$. A test with a reused weight is more informative than a simple chain that never branches.

Check gradients with central differences of an independently written loss expression, changing one input at a time. A very small step amplifies rounding; a very large step measures curvature. The tests use a bounded region and a stated numerical tolerance. Nonfinite inputs are rejected. Finite inputs whose intermediate arithmetic overflows are outside this exercise's representable-arithmetic scope; exact symbolic real-number arithmetic is not promised.

## Interview angle

Explain forward storage, seed the loss gradient with one, and trace one reused input backward. State that contributions add at shared nodes. Follow-ups often ask why reverse mode fits one-loss/many-parameter training, or why a finite-difference mismatch occurs. Discuss reuse of intermediate derivatives for the first, and step size, saturation and omitted paths for the second.

## Connections

Smooth activation derivatives supply the local rules. Dense backprop lifts these scalar edge rules into matrix contractions, while an autograd engine automates reverse traversal and accumulation. The [historical note](../../../sources/library/AI-ML/10-Backpropagation/Rumelhart%20Hinton%20Williams%20Backpropagation%201986.md) connects the calculation to representation learning. End by recording which reused value was hardest to accumulate; that is the exact next retrieval prompt.
