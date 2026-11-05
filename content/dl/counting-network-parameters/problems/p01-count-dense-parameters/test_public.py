from __future__ import annotations

from submission import count_dense_parameters


def test_bias_example():
    assert count_dense_parameters([3, 4, 2]) == ([16, 10], 26), "include one bias per output unit"


def test_no_bias_example():
    assert count_dense_parameters((3, 4, 2), False) == ([12, 8], 20), (
        "bias-disabled count is incorrect"
    )


def test_single_layer():
    assert count_dense_parameters([1, 1]) == ([2], 2), (
        "single layer still includes weights and bias"
    )
