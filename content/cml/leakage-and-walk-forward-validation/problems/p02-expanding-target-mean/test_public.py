import numpy as np
import pytest
from submission import expanding_target_mean


def _encode(categories, target):
    try:
        return np.asarray(expanding_target_mean(categories, target), dtype=np.float64)
    except ValueError:
        raise
    except Exception as exc:
        pytest.fail(f"a valid example raised {type(exc).__name__}: check the running state")


def test_string_categories_use_only_earlier_rows_of_the_same_category():
    got = _encode(["a", "b", "a", "a"], [1.0, 5.0, 3.0, 2.0])
    np.testing.assert_allclose(
        got,
        [np.nan, np.nan, 1.0, 2.0],
        rtol=1e-12,
        equal_nan=True,
        err_msg="row 3 must average the earlier rows of its category only: (1 + 3) / 2 = 2",
    )


def test_integer_categories_average_the_earlier_targets_and_start_with_nan():
    got = _encode([1, 1, 1], [4.0, 6.0, 2.0])
    np.testing.assert_allclose(
        got,
        [np.nan, 4.0, 5.0],
        rtol=1e-12,
        equal_nan=True,
        err_msg="the first row of a category has no earlier rows, so its encoding is NaN",
    )


def test_mismatched_lengths_raise_value_error():
    try:
        expanding_target_mean(["a"], [1.0, 2.0])
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"mismatched lengths raised {type(exc).__name__} instead of ValueError")
    pytest.fail("categories and target of different lengths must raise ValueError")
