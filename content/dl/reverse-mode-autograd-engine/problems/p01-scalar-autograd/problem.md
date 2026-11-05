# Scalar reverse-mode graph

Implement `Value`, a small scalar computation graph supporting `+`, `*`, `.tanh()`,
and `.backward()`. Shared nodes and repeated operand edges must both work.
The exercise ends at scalar reverse mode; no tensor framework is required.

## Contract

- `Value(data)`: a Python int or float, finite and with `abs(data)<=1e6`.
  Invalid scalar values, including complex values and integers too large for
  `float()` such as `10**400`, raise `ValueError`.
- `.data`: read-only Python float. `.grad`: Python float, initially `0.0`.
- `a+b`, `a*b`: both operands are Value nodes; return a new Value with the
  arithmetic result. `.tanh()` returns a new node using stable scalar tanh.
  Every intermediate result must satisfy the same finite scalar bound;
  invalid results raise `ValueError` through construction.
- Graphs are acyclic with at most 100 reachable nodes. No graph mutation or
  changing forward values is supported. Do not mutate an operand's data or grad
  while building a new operation. No implicit numeric operand conversion is required.
- `root.backward()` returns None, resets gradients of all nodes reachable from
  root to zero, seeds root.grad=1.0, and accumulates every path contribution.
  Gradients on unreachable nodes remain unchanged. Calling backward twice on
  the same root yields the same gradients, rather than doubling them.

## Examples

```python
x = Value(3.0)
y = x * x
y.backward()
# y.data -> 9.0; x.grad -> 6.0; y.grad -> 1.0
```

```python
x = Value(2.0)
q = x * x
loss = q + q
loss.backward()
loss.backward()
# loss.data -> 8.0; q.grad -> 2.0; x.grad -> 8.0
```

## Constraints

Allowed: stdlib (math.tanh is suitable) and NumPy. Not allowed: `torch`, `jax`,
`tensorflow`, or `sklearn`. Implement traversal and local derivative rules.
Do not use finite differences inside the implementation. Test tolerance is
atol=1e-8, rtol=1e-6; this accommodates external central-difference checks.

## Done looks like

The examples and constant-root case pass. Hidden families cover shared
subexpressions, duplicate edges, central finite differences of composed functions,
reset behavior across different roots, unreachable graphs, float representation,
saturated tanh, tiny values, and invalid scalars. Every use contributes an edge,
even when two operands refer to the same node.
