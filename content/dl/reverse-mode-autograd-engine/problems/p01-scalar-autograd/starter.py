from __future__ import annotations


class Value:
    """Finite scalar node with addition, multiplication, tanh and reset-on-backward.

    Construct from finite int/float |data|<=1e6, else ValueError. Binary operands
    are Value nodes. Graphs are acyclic with <=100 nodes; intermediate values obey
    the same bound. data is read-only; grad is float. backward resets all reachable
    gradients then seeds this node with 1.0, leaving unreachable nodes unchanged.
    """

    def __init__(self, data: float) -> None:
        raise NotImplementedError

    @property
    def data(self) -> float:
        raise NotImplementedError

    def __add__(self, other: Value) -> Value:
        raise NotImplementedError

    def __mul__(self, other: Value) -> Value:
        raise NotImplementedError

    def tanh(self) -> Value:
        raise NotImplementedError

    def backward(self) -> None:
        raise NotImplementedError
