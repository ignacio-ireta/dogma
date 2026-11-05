from __future__ import annotations

from submission import Value


def test_square_example():
    x = Value(3.0)
    y = x * x
    y.backward()
    assert y.data == 9.0 and x.grad == 6.0


def test_shared_path_example():
    x = Value(2.0)
    q = x * x
    y = q + q
    y.backward()
    assert y.data == 8.0 and x.grad == 8.0


def test_constant_root():
    root = Value(0.0)
    root.backward()
    assert root.grad == 1.0
