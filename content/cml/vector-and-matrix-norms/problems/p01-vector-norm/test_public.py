import pytest
from submission import vector_norm


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_euclidean_norm_of_three_four_is_five():
    assert vector_norm([3, 4], 2) == pytest.approx(5.0, rel=1e-12), (
        "the L2 norm of (3, 4) is sqrt(9 + 16) = 5"
    )


def test_one_norm_is_sum_of_absolute_values():
    assert vector_norm([3, -4], 1) == pytest.approx(7.0, rel=1e-12), (
        "the L1 norm of (3, -4) is |3| + |-4| = 7; a signed sum gives -1"
    )


def test_infinity_norm_is_largest_absolute_entry():
    assert vector_norm([3, -4], float("inf")) == pytest.approx(4.0, rel=1e-12), (
        "the infinity norm of (3, -4) is max(|3|, |-4|) = 4"
    )


def test_p_below_one_raises_value_error():
    _raised_value_error(
        lambda: vector_norm([1, 2], 0.5),
        "p = 0.5 does not define a norm (the triangle inequality fails): raise ValueError",
    )
