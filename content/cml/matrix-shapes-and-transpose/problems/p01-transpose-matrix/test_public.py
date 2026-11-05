import pytest
from submission import transpose


def _must_raise_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_example_two_by_three_becomes_three_by_two():
    assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]], (
        "transpose of a (2, 3) matrix should be (3, 2), with row j equal to column j"
    )


def test_column_vector_becomes_a_single_row():
    assert transpose([[1], [2], [3]]) == [[1, 2, 3]], (
        "a (3, 1) column should transpose to a (1, 3) row, [[1, 2, 3]]; "
        "returning [1, 2, 3] drops the row dimension"
    )


def test_empty_matrix_returns_empty_list():
    assert transpose([]) == [], "the empty matrix is (0, 0), and its transpose is []"


def test_ragged_rows_raise_value_error():
    _must_raise_value_error(
        lambda: transpose([[1, 2, 3], [4, 5]]),
        "ragged rows must raise ValueError; zip(*rows) would silently truncate instead",
    )
