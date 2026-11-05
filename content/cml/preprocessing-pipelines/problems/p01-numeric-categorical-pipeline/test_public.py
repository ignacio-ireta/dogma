import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from submission import fit_transform_pipeline


def _run(train, test, numeric, categorical):
    try:
        return fit_transform_pipeline(train, test, numeric, categorical)
    except Exception as exc:
        pytest.fail(
            f"valid preprocessing request crashed with {type(exc).__name__}: "
            "check missing-value fills and unseen categories"
        )


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_worked_example_fits_train_and_only_transforms_test():
    train = pd.DataFrame(
        {"age": [20.0, np.nan, 40.0, 30.0], "city": ["north", np.nan, "south", "north"]}
    )
    test = pd.DataFrame({"age": [np.nan, 50.0], "city": ["west", np.nan]})
    train_out, test_out = _run(train, test, ["age"], ["city"])
    np.testing.assert_allclose(
        train_out,
        [[-np.sqrt(2), 1, 0], [0, 1, 0], [np.sqrt(2), 0, 1], [0, 1, 0]],
        atol=1e-9,
        err_msg="train output differs: median-fill and mode-fill from train, then standardize",
    )
    np.testing.assert_allclose(
        test_out,
        [[0, 0, 0], [20 / np.sqrt(50), 1, 0]],
        atol=1e-9,
        err_msg="test output differs: use the training mean, scale and categories only",
    )


def test_mode_tie_goes_to_the_smallest_value():
    train = pd.DataFrame({"plan": ["pro", "basic", "pro", "basic", np.nan]})
    test = pd.DataFrame({"plan": [np.nan, "pro"]})
    train_out, test_out = _run(train, test, [], ["plan"])
    if train_out[4].tolist() != [1.0, 0.0]:
        pytest.fail("a tied mode must resolve to the smallest value, 'basic'")
    if test_out.tolist() != [[1.0, 0.0], [0.0, 1.0]]:
        pytest.fail("test values: a missing test value takes the training mode")


def test_a_name_missing_from_the_test_frame_raises_value_error():
    train = pd.DataFrame({"age": [1.0, 2.0]})
    test = pd.DataFrame({"income": [1.0]})
    _expect_value_error(
        lambda: fit_transform_pipeline(train, test, ["age"], []),
        "a column missing from the test frame was accepted: check both frames",
    )
