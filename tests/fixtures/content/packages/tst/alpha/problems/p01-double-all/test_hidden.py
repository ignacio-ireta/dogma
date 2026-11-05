import numpy as np
from submission import double_all


def test_matches_numpy_on_random_input() -> None:
    values = np.random.default_rng(0).normal(size=20)
    np.testing.assert_allclose(
        double_all(values.tolist()), 2.0 * values, err_msg="result differs from NumPy doubling"
    )


def test_integer_input_returns_floats() -> None:
    assert all(isinstance(value, float) for value in double_all([1, 2]))


def test_large_magnitudes_double_without_overflow() -> None:
    assert double_all([1e300]) == [2e300]


def test_tiny_magnitudes_are_kept() -> None:
    assert double_all([1e-300]) == [2e-300]


def test_input_list_is_not_mutated() -> None:
    values = [1.0, 2.0]
    double_all(values)
    assert values == [1.0, 2.0]
