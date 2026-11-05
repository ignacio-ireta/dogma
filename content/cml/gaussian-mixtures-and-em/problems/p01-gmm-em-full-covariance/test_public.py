import numpy as np
import pytest
from submission import gmm_em


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError as exc:
        return exc
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead: {exc}")
    pytest.fail(mistake)


def test_initial_state_of_the_two_point_fixture_has_equal_weights_and_closed_form_loglik():
    w, _, _, lls = gmm_em([[0.0], [2.0]], 2, [[0.0], [2.0]], 0, 1e-3)
    np.testing.assert_allclose(
        w,
        [0.5, 0.5],
        atol=1e-12,
        err_msg="initial weights must be 1/k each",
    )
    expected = 2 * (np.log(0.5) - 0.5 * np.log(2 * np.pi) + np.log1p(np.exp(-2.0)))
    np.testing.assert_allclose(
        lls[0],
        expected,
        rtol=1e-10,
        err_msg="initial log-likelihood differs: check the density's 2*pi and the mixture sum",
    )


def test_one_iteration_moves_each_mean_to_its_responsibility_weighted_mean():
    _, mu, cov, _ = gmm_em([[0.0], [2.0]], 2, [[0.0], [2.0]], 1, 1e-3)
    np.testing.assert_allclose(
        mu.ravel(),
        [0.2384, 1.7616],
        atol=1e-4,
        err_msg="means after one iteration: the M-step should use responsibility-weighted means",
    )
    np.testing.assert_allclose(
        cov.ravel(),
        [0.4200, 0.4200],
        atol=1e-4,
        err_msg="covariances after one iteration: use the weighted scatter about the new mean",
    )


def test_more_components_than_samples_raises_value_error():
    _raised_value_error(
        lambda: gmm_em([[0.0], [2.0]], 3, [[0.0], [2.0], [5.0]], 1, 1e-3),
        "k larger than the number of samples must raise ValueError",
    )
