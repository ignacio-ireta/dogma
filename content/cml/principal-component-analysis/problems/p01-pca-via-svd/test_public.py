import numpy as np
import pytest
from submission import pca


def _call(X, k):
    try:
        return pca(X, k)
    except Exception as exc:
        pytest.fail(
            f"pca raised {type(exc).__name__} on a valid input: check that components is (k, d) "
            "and transformed is (n, k), built from the SVD's right singular vectors"
        )


def _field(result, key):
    if not isinstance(result, dict) or key not in result:
        pytest.fail(f"the result has no {key!r} entry: return a dict with exactly the four keys")
    return np.asarray(result[key], dtype=np.float64)


def _check_close(actual, expected, mistake):
    if actual.shape != np.shape(expected) or not np.allclose(
        actual, expected, rtol=1e-9, atol=1e-9
    ):
        pytest.fail(mistake)


def test_worked_example_gives_hand_computed_axis_variance_and_ratio():
    result = _call([[-3, -1], [0, -2], [0, 3]], 1)
    _check_close(
        _field(result, "components"),
        [[1 / np.sqrt(10), 3 / np.sqrt(10)]],
        "components are wrong: the first axis is (1, 3)/sqrt(10), with its largest entry positive",
    )
    _check_close(
        _field(result, "explained_variance"),
        [7.5],
        "explained variance is wrong: it is the sample variance (divisor n - 1) of the projection",
    )
    _check_close(
        _field(result, "explained_variance_ratio"),
        [0.75],
        "explained-variance ratio is wrong: divide by the total variance of all columns (10 here)",
    )
    _check_close(
        _field(result, "transformed"),
        [[-5 / np.sqrt(10)], [-5 / np.sqrt(10)], [10 / np.sqrt(10)]],
        "transformed scores are wrong: centre the columns, then project onto the signed axis",
    )


def test_tied_entries_make_the_lowest_index_positive():
    result = _call([[1, -1], [-1, 1], [0, 0]], 1)
    _check_close(
        _field(result, "components"),
        [[1 / np.sqrt(2), -1 / np.sqrt(2)]],
        "sign rule: with a tie between entries, the lowest index must be the positive one",
    )


def test_n_components_above_min_of_rows_and_columns_raises_value_error():
    try:
        pca([[1, 2], [3, 4]], 3)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"too many components raised {type(exc).__name__}; raise ValueError")
    pytest.fail("n_components above min(n, d) must raise ValueError")


def test_constant_columns_give_zero_ratios_not_nan():
    result = _call([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]], 1)
    ratio = _field(result, "explained_variance_ratio")
    if np.isnan(ratio).any():
        pytest.fail("constant data gave NaN ratios: a zero total variance must give 0.0")
    _check_close(ratio, [0.0], "constant data must give ratios of 0.0 (no variance to explain)")
