import re

import pytest
from submission import matmul


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def _names_shape(message, shape):
    return re.search(rf"\({shape[0]},\s*{shape[1]}\)", message) is not None


def test_example_two_by_three_times_three_by_two():
    assert matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]) == [[58, 64], [139, 154]], (
        "product of (2, 3) and (3, 2) should be (2, 2); entry (0, 0) is 1*7 + 2*9 + 3*11 = 58"
    )


def test_identity_on_the_left_leaves_the_matrix_unchanged():
    B = [[5, 6], [7, 8]]
    assert matmul([[1, 0], [0, 1]], B) == B, (
        "multiplying by the identity should return the other matrix unchanged"
    )


def test_single_entry_product_is_a_plain_multiplication():
    assert matmul([[2]], [[3]]) == [[6]], "a (1, 1) times (1, 1) product is just the scalar product"


def test_inner_dimension_mismatch_raises_and_names_both_shapes():
    exc = _raised_value_error(
        lambda: matmul([[1, 2, 3], [4, 5, 6]], [[1, 2], [3, 4]]),
        "inner dimensions 3 and 2 differ, so matmul must raise ValueError",
    )
    message = str(exc)
    assert _names_shape(message, (2, 3)) and _names_shape(message, (2, 2)), (
        f"the ValueError should name both shapes, like (2, 3) @ (2, 2); got: {message}"
    )
