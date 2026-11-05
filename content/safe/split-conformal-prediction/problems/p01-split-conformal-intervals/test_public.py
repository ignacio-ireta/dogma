# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import numpy as np
from submission import split_conformal_interval

CAL_PRED = [0, 0, 0, 0, 0, 0, 0, 0, 0]
CAL_TRUE = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9]


def test_threshold_is_the_eighth_smallest_residual_at_alpha_point_two():
    lower, upper = split_conformal_interval(CAL_PRED, CAL_TRUE, [5.0], 0.2)
    np.testing.assert_allclose(
        lower, [4.2], atol=1e-12, err_msg="lower bound: subtract the 8th smallest |residual| (0.8)"
    )
    np.testing.assert_allclose(
        upper,
        [5.8],
        atol=1e-12,
        err_msg="upper bound: add the same threshold, symmetric around the prediction",
    )


def test_rank_above_calibration_size_gives_infinite_interval():
    lower, upper = split_conformal_interval(CAL_PRED, CAL_TRUE, [5.0], 0.05)
    assert lower[0] == -np.inf and upper[0] == np.inf, (
        "k = ceil(10 * 0.95) = 10 exceeds n = 9, so the threshold must be +inf"
    )


def test_single_calibration_point_gives_its_residual_as_threshold():
    lower, upper = split_conformal_interval([2.0], [3.5], [10.0], 0.5)
    np.testing.assert_allclose(
        [lower[0], upper[0]],
        [8.5, 11.5],
        atol=1e-12,
        err_msg="n = 1, alpha = 0.5: k = 1, so the threshold is the one residual 1.5",
    )
