from __future__ import annotations

import math


class Value:
    """Wrong on purpose: traversal has no visited set, so shared nodes are counted once per path.

    data must be finite int/float with |data|<=1e6; invalid values raise ValueError.
    Binary operands are Value nodes; acyclic graphs contain <=100 nodes. data is
    read-only. Each backward resets reachable grads and leaves other nodes alone.
    """

    def __init__(self, data: float) -> None:
        if not isinstance(data, (int, float)) or abs(data) > 1e6 or not math.isfinite(data):
            raise ValueError("data must be a bounded finite real scalar")
        self._data = float(data)
        self.grad = 0.0
        # Keep edges, including duplicates, rather than deduplicating operands.
        self._edges: tuple[tuple[Value, float], ...] = ()

    @property
    def data(self) -> float:
        """Read-only scalar forward value."""
        return self._data

    def __add__(self, other: Value) -> Value:
        """Create an addition node with local partials equal to one."""
        result = Value(self.data + other.data)
        result._edges = ((self, 1.0), (other, 1.0))
        return result

    def __mul__(self, other: Value) -> Value:
        """Create a multiplication node with the opposite operand as partial."""
        result = Value(self.data * other.data)
        result._edges = ((self, other.data), (other, self.data))
        return result

    def tanh(self) -> Value:
        """Create a numerically stable tanh node using its cached output."""
        result = Value(math.tanh(self.data))
        result._edges = ((self, 1 - result.data**2),)
        return result

    def backward(self) -> None:
        """Reset reachable gradients, seed the root, and traverse parents last."""
        order: list[Value] = []
        visited: set[Value] = set()  # noqa: F841

        def visit(node: Value) -> None:
            for parent, _ in node._edges:
                visit(parent)
            order.append(node)

        visit(self)
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        # Nodes are visited once, but every incoming contribution is added.
        for node in reversed(order):
            for parent, partial in node._edges:
                parent.grad += node.grad * partial
