import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from submission import gmm_em


def _close(actual, desired, *, err_msg, **tolerance):
    """assert_allclose whose failure shows err_msg as the first line, without the values."""
    try:
        np.testing.assert_allclose(actual, desired, **tolerance)
    except AssertionError:
        pytest.fail(err_msg, pytrace=False)


def _equal(actual, desired, *, err_msg):
    """assert_array_equal whose failure shows err_msg as the first line, without the values."""
    try:
        np.testing.assert_array_equal(actual, desired)
    except AssertionError:
        pytest.fail(err_msg, pytrace=False)


def _call(X, k, init, n_iter, reg):
    """Run the learner's function; a crash on valid input is a contract failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = gmm_em(X, k, init, n_iter, reg)
    except RuntimeWarning:
        pytest.fail("a NumPy warning was raised on valid input: guard log(0) and division by N_k")
    except Exception as exc:
        pytest.fail(f"valid input must be accepted; the call raised {type(exc).__name__}")
    if not (isinstance(result, tuple) and len(result) == 4):
        pytest.fail("return a four-item tuple (weights, means, covs, lls)")
    return result


def _floor(S, reg):
    values, vectors = np.linalg.eigh(S)
    return vectors @ np.diag(np.maximum(values, reg)) @ vectors.T


def _ref_log_terms(X, w, means, covs):
    """Independent log(pi_j) + log N(x_i | mu_j, Sigma_j) via SciPy's density."""
    columns = []
    for j in range(len(w)):
        if w[j] == 0.0:
            columns.append(np.full(len(X), -np.inf))
        else:
            dist = multivariate_normal(mean=means[j], cov=covs[j])
            columns.append(np.log(w[j]) + dist.logpdf(X))
    return np.column_stack(columns)


def _ref_e_step(X, w, means, covs):
    log_terms = _ref_log_terms(X, w, means, covs)
    lse = logsumexp(log_terms, axis=1)
    return np.exp(log_terms - lse[:, None]), float(lse.sum())


def _reference(X, k, init_means, n_iter, reg):
    """Plain-loop EM written independently of the learner's code."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    weights = np.full(k, 1.0 / k)
    means = np.asarray(init_means, dtype=np.float64).copy()
    centred = X - X.mean(axis=0)
    pooled = centred.T @ centred / n
    covs = np.stack([_floor(pooled, reg) for _ in range(k)])
    lls = [_ref_e_step(X, weights, means, covs)[1]]
    for _ in range(n_iter):
        resp, _ = _ref_e_step(X, weights, means, covs)
        counts = resp.sum(axis=0)
        new_w = np.zeros(k)
        new_m = means.copy()
        new_c = covs.copy()
        for j in range(k):
            if counts[j] == 0.0:
                continue
            new_w[j] = counts[j] / n
            new_m[j] = (resp[:, j, None] * X).sum(axis=0) / counts[j]
            diff = X - new_m[j]
            scatter = np.einsum("i,ia,ib->ab", resp[:, j], diff, diff) / counts[j]
            new_c[j] = _floor(scatter, reg)
        weights, means, covs = new_w, new_m, new_c
        lls.append(_ref_e_step(X, weights, means, covs)[1])
    return weights, means, covs, np.array(lls)


def _blobs(seed, n=60):
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [4.0, 1.0], [1.0, 5.0]])
    return np.vstack([c + rng.normal(scale=1.0, size=(n // 3, 2)) for c in centres])


def _raised_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


def test_two_point_fixture_one_iteration_matches_hand_computed_values():
    # Responsibility of component 1 for x = 0 is r = 1 / (1 + e^-2), and the same r for the
    # other component at x = 2 with the roles swapped. The updated means are then
    # 2(1 - r) and 2r, and both covariances equal 4r(1 - r). Weights stay 1/2.
    r = 1.0 / (1.0 + np.exp(-2.0))
    mu1, mu2, var = 2.0 * (1.0 - r), 2.0 * r, 4.0 * r * (1.0 - r)
    w, mu, cov, lls = _call([[0.0], [2.0]], 2, [[0.0], [2.0]], 1, 1e-3)
    _close(w, [0.5, 0.5], rtol=1e-12, err_msg="weights must stay 1/2 each")
    _close(
        mu.ravel(),
        [mu1, mu2],
        rtol=1e-10,
        err_msg="means do not match the responsibility-weighted means of the fixture",
    )
    _close(
        cov.reshape(2, -1).ravel(),
        [var, var],
        rtol=1e-10,
        err_msg="covariances do not match the weighted scatter about the updated means",
    )
    # Log-likelihood after one iteration, from the 1-D mixture density directly.
    sd = np.sqrt(var)
    xs = np.array([0.0, 2.0])
    dens = 0.5 * (
        np.exp(-((xs[:, None] - np.array([mu1, mu2])) ** 2) / (2 * var)) / (np.sqrt(2 * np.pi) * sd)
    ).sum(axis=1)
    _close(
        lls[1],
        np.sum(np.log(dens)),
        rtol=1e-10,
        err_msg="log-likelihood after one iteration does not match the mixture density",
    )


def test_initial_log_likelihood_matches_closed_form():
    _, _, _, lls = _call([[0.0], [2.0]], 2, [[0.0], [2.0]], 0, 1e-3)
    assert lls.shape == (1,), "with n_iter = 0, lls must contain only the initial log-likelihood"
    expected = 2 * (np.log(0.5) - 0.5 * np.log(2 * np.pi) + np.log1p(np.exp(-2.0)))
    _close(
        lls[0],
        expected,
        rtol=1e-10,
        err_msg="initial log-likelihood: the density or the mixture sum is wrong",
    )


def test_matches_independent_reference_on_seeded_two_dimensional_data():
    X = _blobs(5)
    init = X[[0, 20, 40]]
    w, mu, cov, lls = _call(X, 3, init, 25, 1e-3)
    want_w, want_mu, want_cov, want_lls = _reference(X, 3, init, 25, 1e-3)
    _close(w, want_w, rtol=1e-8, err_msg="weights differ from the reference: check N_k / n")
    _close(mu, want_mu, rtol=1e-8, atol=1e-10, err_msg="means differ: check the weighted mean")
    _close(
        cov,
        want_cov,
        rtol=1e-8,
        atol=1e-10,
        err_msg="covariances differ: check the weighted scatter and its floor",
    )
    _close(
        lls,
        want_lls,
        rtol=1e-9,
        err_msg="log-likelihood sequence differs: check the density and the mixture sum",
    )


def test_log_likelihood_never_decreases_across_iterations():
    for seed in range(10):
        X = _blobs(100 + seed, n=45) * (1.0 + seed / 4.0)
        init = X[[0, 15, 30]]
        _, _, _, lls = _call(X, 3, init, 40, 1e-3)
        drops = np.diff(lls)
        tolerance = 1e-9 * np.maximum(1.0, np.abs(lls[:-1]))
        assert np.all(drops >= -tolerance), (
            "log-likelihood decreased between iterations: EM must not lower it, so check the "
            "M-step (weighted means and scatter) and the floor"
        )


def test_covariance_floor_holds_when_points_collapse_onto_one_location():
    # Three identical samples can capture a component that then has zero spread.
    X = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [5.0, 1.0], [6.0, 0.0], [5.0, -1.0]])
    reg = 1e-3
    _, _, cov, lls = _call(X, 2, [[0.0, 0.0], [5.0, 0.0]], 15, reg)
    assert np.all(np.isfinite(lls)), "a collapsed covariance made the log-likelihood non-finite"
    for j, c in enumerate(cov):
        smallest = np.linalg.eigvalsh(c).min()
        assert smallest >= reg * (1 - 1e-9), (
            f"covariance {j} fell below the floor: raise every eigenvalue to at least reg"
        )


def test_far_apart_points_are_handled_in_log_space():
    # Every sample is about 1400 away from both initial means, so each density is exp(-1e8),
    # which underflows to 0 in float64. A naive ratio then gives 0 / 0 = nan for every sample.
    X = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.2], [-0.1, 0.1]])
    init = [[-1000.0, -1000.0], [1000.0, 1000.0]]
    _, mu, _, lls = _call(X, 2, init, 2, 0.1)
    assert np.all(np.isfinite(lls)) and np.all(np.isfinite(mu)), (
        "far-apart samples gave non-finite results: compute responsibilities with log-sum-exp"
    )
    want = _reference(X, 2, init, 2, 0.1)
    _close(mu, want[1], rtol=1e-7, err_msg="means differ on far-apart samples: check the E-step")


def test_collinear_samples_start_from_a_floored_covariance():
    # The pooled covariance of points on one line is singular, so the initial covariance must be
    # floored too, or the first E-step cannot factor it.
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    reg = 1e-2
    _, _, cov, _ = _call(X, 2, [[0.0, 0.0], [4.0, 4.0]], 0, reg)  # n_iter = 0: the start itself
    for j, c in enumerate(cov):
        assert np.linalg.eigvalsh(c).min() >= reg * (1 - 1e-9), (
            f"initial covariance {j} fell below the floor: floor the pooled covariance at the start"
        )


def test_component_with_no_responsibility_keeps_its_mean_and_gets_zero_weight():
    X = np.array([[0.0], [0.5], [1.0], [0.2]])
    reg = 1e-3
    init = [[0.5], [1.0e6]]  # the second mean is so far away that it takes no responsibility
    w, mu, _, lls = _call(X, 2, init, 3, reg)
    assert w[1] == 0.0, "a component with no responsibility must end with weight 0"
    assert mu[1, 0] == 1.0e6, "a dead component must keep its previous mean, not move"
    assert np.isclose(w.sum(), 1.0), "weights must still sum to 1"
    assert np.all(np.isfinite(lls)), "a dead component made the log-likelihood non-finite"


@settings(max_examples=25, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=0, max_value=10_000))
def test_sample_order_does_not_change_the_fit(seed):
    X = _blobs(seed % 50, n=30)
    init = X[[0, 10, 20]]
    base = _call(X, 3, init, 10, 1e-3)
    order = np.random.default_rng(seed).permutation(len(X))
    permuted = _call(X[order], 3, init, 10, 1e-3)
    for name, a, b in zip(("weights", "means", "covs", "lls"), base, permuted, strict=True):
        _close(
            a,
            b,
            rtol=1e-8,
            atol=1e-10,
            err_msg=f"{name} changed when samples were reordered: EM treats samples as a set",
        )


def test_common_offset_of_data_and_means_leaves_the_fit_unchanged():
    X = _blobs(7, n=30)
    init = X[[0, 10, 20]]
    w0, mu0, cov0, lls0 = _call(X, 3, init, 10, 1e-3)
    offset = 1.0e6
    w1, mu1, cov1, lls1 = _call(X + offset, 3, init + offset, 10, 1e-3)
    _close(w1, w0, rtol=1e-8, err_msg="a common offset changed the weights")
    _close(
        mu1 - offset,
        mu0,
        atol=1e-6,
        err_msg="a common offset moved the means: compute X - mu directly, not |x|^2 expansions",
    )
    _close(cov1, cov0, atol=1e-6, err_msg="a common offset changed the covariances beyond rounding")
    _close(lls1, lls0, rtol=1e-8, err_msg="a common offset changed the lls")


def test_outputs_have_documented_shapes_and_dtypes():
    X = _blobs(3, n=18)
    w, mu, cov, lls = _call(X.astype(np.float32), 3, X[[0, 6, 12]], 4, 1e-3)
    assert np.asarray(w).shape == (3,), "shape mismatch: weights must have shape (k,)"
    assert np.asarray(mu).shape == (3, 2), "shape mismatch: means must have shape (k, d)"
    assert np.asarray(cov).shape == (3, 2, 2), "shape mismatch: covs must have shape (k, d, d)"
    assert np.asarray(lls).shape == (5,), "shape mismatch: lls must have n_iter + 1 entries"
    for name, arr in (("weights", w), ("means", mu), ("covs", cov), ("lls", lls)):
        assert np.asarray(arr).dtype == np.float64, f"{name} must be float64 even for float32 input"
    _close(
        np.asarray(cov),
        np.transpose(np.asarray(cov), (0, 2, 1)),
        atol=1e-12,
        err_msg="covariances must be symmetric",
    )


INVALID = [
    ("zero-components", [[0.0], [2.0]], 0, [[0.0]], 1, 1e-3, "k must be at least 1"),
    (
        "more-components-than-samples",
        [[0.0], [2.0]],
        3,
        [[0.0], [2.0], [5.0]],
        1,
        1e-3,
        "k must be at most the number of samples",
    ),
    (
        "init-count-mismatch",
        [[0.0], [2.0]],
        2,
        [[0.0]],
        1,
        1e-3,
        "init_means must have exactly k rows",
    ),
    (
        "init-dimension-mismatch",
        [[0.0], [2.0]],
        2,
        [[0.0, 0.0], [2.0, 0.0]],
        1,
        1e-3,
        "init_means must have as many columns as X",
    ),
    (
        "one-dimensional-X",
        [0.0, 2.0],
        2,
        [[0.0], [2.0]],
        1,
        1e-3,
        "X must be a 2-D array of samples",
    ),
    (
        "nan-sample",
        [[0.0], [float("nan")]],
        2,
        [[0.0], [2.0]],
        1,
        1e-3,
        "non-finite samples must raise ValueError",
    ),
    (
        "negative-iterations",
        [[0.0], [2.0]],
        2,
        [[0.0], [2.0]],
        -1,
        1e-3,
        "n_iter must be non-negative",
    ),
    (
        "boolean-iterations",
        [[0.0], [2.0]],
        2,
        [[0.0], [2.0]],
        True,
        1e-3,
        "a boolean is not an iteration count: raise ValueError",
    ),
    ("zero-floor", [[0.0], [2.0]], 2, [[0.0], [2.0]], 1, 0.0, "reg must be strictly positive"),
    ("negative-floor", [[0.0], [2.0]], 2, [[0.0], [2.0]], 1, -1.0, "reg must be strictly positive"),
    (
        "nan-floor",
        [[0.0], [2.0]],
        2,
        [[0.0], [2.0]],
        1,
        float("nan"),
        "reg must be a finite number",
    ),
]


@pytest.mark.parametrize("case", INVALID, ids=[case[0] for case in INVALID])
def test_invalid_inputs_raise_value_error(case):
    _name, X, k, init, n_iter, reg, mistake = case
    _raised_value_error(lambda: gmm_em(X, k, init, n_iter, reg), mistake)


def test_inputs_are_not_mutated():
    X = _blobs(9, n=21)
    init = np.array(X[[0, 7, 14]], dtype=np.float64)
    X0, init0 = X.copy(), init.copy()
    _call(X, 3, init, 6, 1e-3)
    _equal(X, X0, err_msg="samples X were modified in place")
    _equal(init, init0, err_msg="init_means was modified in place")


def test_zero_dimensional_samples_raise_value_error():
    _raised_value_error(
        lambda: gmm_em(np.zeros((2, 0)), 2, np.zeros((2, 0)), 1, 1e-3),
        "samples need at least one column: a shape with zero columns must raise ValueError",
    )


def test_zero_components_raise_value_error_even_with_an_empty_init():
    _raised_value_error(
        lambda: gmm_em([[0.0], [2.0]], 0, np.zeros((0, 1)), 0, 1e-3),
        "k must be at least 1: a zero component count must raise ValueError",
    )


def test_live_component_after_a_dead_one_still_takes_its_m_step():
    # The first component receives no responsibility, so it is dead. The second must still update.
    X = np.array([[0.0], [0.5], [1.0], [0.2]])
    w, mu, _, _ = _call(X, 2, [[1.0e6], [0.5]], 1, 1e-3)
    assert w[0] == 0.0, "a dead component must end with weight 0"
    assert np.isclose(w[1], 1.0), "the live component must keep all the weight"
    _close(
        mu[1, 0],
        X[:, 0].mean(),
        rtol=1e-10,
        err_msg="a live component after a dead one was not updated: skip only the dead component",
    )


def test_initial_covariance_is_the_pooled_scatter_when_no_eigenvalue_is_floored():
    # With reg tiny, no eigenvalue is raised, so the floor must return the pooled scatter itself.
    rng = np.random.default_rng(11)
    X = rng.normal(size=(12, 3)) * np.array([1.0, 2.0, 0.5])
    _, _, cov, _ = _call(X, 2, X[[0, 6]], 0, 1e-12)
    centred = X - X.mean(axis=0)
    scatter = centred.T @ centred / len(X)
    for j, c in enumerate(cov):
        _close(
            c,
            scatter,
            rtol=1e-9,
            atol=1e-12,
            err_msg=f"covariance {j} is not the pooled scatter: the floor must return V "
            f"diag(max) V'",
        )


def test_non_finite_init_means_raise_value_error():
    # Checked with n_iter = 0, so the result comes from validation alone, not later numerics.
    # One non-finite entry among finite ones still violates the contract.
    for init in ([[0.0], [float("nan")]], [[float("inf")], [2.0]]):
        _raised_value_error(
            lambda init=init: gmm_em([[0.0], [2.0]], 2, init, 0, 1e-3),
            "init_means must be finite: a non-finite entry must raise ValueError",
        )
