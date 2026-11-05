import pytest
from submission import leaf_weight_and_split_gain


def test_worked_example_gives_leaf_weights_and_gain():
    w_left, w_right, gain = leaf_weight_and_split_gain(
        [-2.0, -1.0], [1.0, 1.0], [2.0], [1.0], reg_lambda=1.0, gamma=0.5
    )
    assert w_left == pytest.approx(1.0, abs=1e-12), (
        "left leaf weight is wrong: use w = -G / (H + reg_lambda) on the left side"
    )
    assert w_right == pytest.approx(-1.0, abs=1e-12), (
        "right leaf weight is wrong: use w = -G / (H + reg_lambda) on the right side"
    )
    assert gain == pytest.approx(1.875, abs=1e-12), (
        "split gain is wrong: check the 1/2 factor, the parent term and the gamma term"
    )


def test_opposite_gradients_give_opposite_leaf_weights():
    w_left, w_right, gain = leaf_weight_and_split_gain(
        [1.0], [1.0], [-1.0], [1.0], reg_lambda=1.0, gamma=0.0
    )
    assert (w_left, w_right) == pytest.approx((-0.5, 0.5), abs=1e-12), (
        "leaf weights have the wrong sign or scale: weight is minus G over (H + lambda)"
    )
    assert gain == pytest.approx(0.5, abs=1e-12), (
        "split gain is wrong for symmetric sides: the parent term should vanish here"
    )


def test_empty_right_side_with_positive_lambda_gives_zero_weight_and_gain_minus_gamma():
    w_left, w_right, gain = leaf_weight_and_split_gain(
        [2.0], [1.0], [], [], reg_lambda=1.0, gamma=0.25
    )
    assert w_left == pytest.approx(-1.0, abs=1e-12), (
        "left leaf weight is wrong for a one-sided split"
    )
    assert w_right == pytest.approx(0.0, abs=1e-12), (
        "an empty side has no gradient sum, so its leaf weight must be zero"
    )
    assert gain == pytest.approx(-0.25, abs=1e-12), (
        "a split with an empty side gains nothing: the gain must be exactly minus gamma"
    )


def test_negative_reg_lambda_raises_value_error():
    # Hessians sum to 2, so H + reg_lambda is still positive: only the regulariser rule can fail.
    try:
        leaf_weight_and_split_gain([1.0], [2.0], [1.0], [2.0], reg_lambda=-1.0, gamma=0.0)
    except ValueError:
        return
    pytest.fail("a negative reg_lambda must raise ValueError, not return a split")
