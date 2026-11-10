from __future__ import annotations

import math


def _to_scalar(value: object) -> float:
    if isinstance(value, complex):
        raise ValueError("Value data must be a real number")
    try:
        v = float(value)  # type: ignore[arg-type]
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("Value data must be a finite real number") from exc
    if math.isnan(v) or math.isinf(v) or abs(v) > 1e6:
        raise ValueError("Value data must be finite with absolute value <= 1e6")
    return v


class Value:
    """Finite scalar node with addition, multiplication, tanh and reset-on-backward.

    Construct from finite int/float |data|<=1e6, else ValueError. Binary operands
    are Value nodes. Graphs are acyclic with <=100 nodes; intermediate values obey
    the same bound. data is read-only; grad is float. backward resets all reachable
    gradients then seeds this node with 1.0, leaving unreachable nodes unchanged.
    """

    def __init__(self, data: float) -> None:
        self._data = _to_scalar(data)
        self.grad = 0.0
        self._prev: tuple[Value, ...] = ()
        self._backward_fn = None

    @property
    def data(self) -> float:
        return self._data

    @staticmethod
    def _wrap(other: object) -> Value:
        if not isinstance(other, Value):
            raise TypeError("operands must be Value nodes")
        return other

    def __add__(self, other: Value) -> Value:
        other = Value._wrap(other)
        out = Value(self._data + other._data)
        out._prev = (self, other)

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward_fn = _backward
        return out

    def __mul__(self, other: Value) -> Value:
        other = Value._wrap(other)
        out = Value(self._data * other._data)
        out._prev = (self, other)

        def _backward() -> None:
            self.grad += other._data * out.grad
            other.grad += self._data * out.grad

        out._backward_fn = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self._data)
        out = Value(t)
        out._prev = (self,)

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward_fn = _backward
        return out

    def backward(self) -> None:
        # Topological order by iterative depth-first search.
        order: list[Value] = []
        visited: set[int] = set()
        stack: list[tuple[Value, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for child in node._prev:
                if id(child) not in visited:
                    stack.append((child, False))

        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            if node._backward_fn is not None:
                node._backward_fn()
