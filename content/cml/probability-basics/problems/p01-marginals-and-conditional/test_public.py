import numpy as np
from submission import marginals_and_conditional


def test_perfectly_dependent_pair_has_identity_conditional():
    p_x, p_y, cond = marginals_and_conditional([[0.5, 0.0], [0.0, 0.5]])
    np.testing.assert_allclose(p_x, [0.5, 0.5], atol=1e-12, err_msg="p_x must be the row sums")
    np.testing.assert_allclose(p_y, [0.5, 0.5], atol=1e-12, err_msg="p_y must be the column sums")
    np.testing.assert_allclose(
        cond,
        np.eye(2),
        atol=1e-12,
        err_msg="knowing X = i fixes Y = i, so P(Y | X) must be the identity matrix",
    )


def test_weather_and_traffic_table_gives_expected_marginals_and_conditionals():
    joint = [[0.20, 0.20, 0.10], [0.05, 0.15, 0.30]]
    p_x, p_y, cond = marginals_and_conditional(joint)
    np.testing.assert_allclose(p_x, [0.5, 0.5], atol=1e-12, err_msg="p_x must be the row sums")
    np.testing.assert_allclose(
        p_y, [0.25, 0.35, 0.40], atol=1e-12, err_msg="p_y must be the column sums"
    )
    np.testing.assert_allclose(
        cond,
        [[0.4, 0.4, 0.2], [0.1, 0.3, 0.6]],
        atol=1e-12,
        err_msg="each row must be divided by its row sum, not by a column sum",
    )


def test_zero_probability_row_gives_nan_conditional_row():
    _, _, cond = marginals_and_conditional([[0.0, 0.0], [0.5, 0.5]])
    assert np.isnan(cond[0]).all(), (
        "P(Y | X=0) is undefined when P(X=0) = 0: return NaN for that row, not a made-up row"
    )
    np.testing.assert_allclose(cond[1], [0.5, 0.5], atol=1e-12)


def test_table_that_sums_to_two_raises_value_error():
    try:
        marginals_and_conditional([[0.5, 0.5], [0.5, 0.5]])
    except ValueError:
        return
    raise AssertionError("a table that sums to 2 is not a joint distribution: raise ValueError")
