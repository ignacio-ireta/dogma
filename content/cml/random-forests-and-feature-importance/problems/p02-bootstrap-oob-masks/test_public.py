import numpy as np
import pytest
from submission import bootstrap_oob


def _call(n, n_trees, seed):
    """Call the function; a crash is reported as the broken contract rule, not as a traceback."""
    try:
        return bootstrap_oob(n, n_trees, seed)
    except Exception as exc:
        pytest.fail(
            f"a valid request must return (in_bag, oob); the call raised "
            f"{type(exc).__name__}: {exc}"
        )


def test_each_tree_draws_n_rows_with_replacement():
    in_bag, _ = _call(4, 2, 0)
    assert in_bag.shape == (2, 4), "shape mismatch: one row of counts per tree, one column per row"
    np.testing.assert_array_equal(
        in_bag.sum(axis=1),
        np.array([4, 4]),
        err_msg="each tree must draw exactly n rows, so its counts sum to n",
    )


def test_oob_mask_is_exactly_the_rows_never_drawn():
    in_bag, oob = _call(4, 2, 0)
    np.testing.assert_array_equal(
        oob,
        in_bag == 0,
        err_msg="the out-of-bag mask must be True exactly where the in-bag count is zero",
    )
    assert oob.dtype == bool and np.issubdtype(in_bag.dtype, np.integer), (
        "the mask must be boolean and the counts must be integers"
    )


def test_single_row_is_always_in_bag():
    _, oob = _call(1, 3, 7)
    np.testing.assert_array_equal(
        oob,
        np.zeros((3, 1), dtype=bool),
        err_msg="with one row, every bootstrap sample contains it: nothing is out of bag",
    )


@pytest.mark.parametrize(
    ("n", "n_trees", "seed"),
    [(0, 2, 0), (3, 0, 0), (3, 2, -1)],
    ids=["no-rows", "no-trees", "negative-seed"],
)
def test_invalid_arguments_raise_value_error(n, n_trees, seed):
    try:
        bootstrap_oob(n, n_trees, seed)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid arguments must raise ValueError; raised {type(exc).__name__}")
    pytest.fail("invalid arguments must raise ValueError, not return samples")
