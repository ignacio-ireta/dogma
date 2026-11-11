import numpy as np
import pytest
from submission import bootstrap_percentile_ci

from dogma_data import load

# Cases for the protocol comparison: (seed, n, confidence, statistic name).
PROTOCOL_CASES = [
    (0, 5, 0.95, "mean"),
    (1, 12, 0.90, "mean"),
    (2, 30, 0.95, "median"),
    (3, 7, 0.80, "mean"),
    (4, 50, 0.99, "median"),
    (5, 9, 0.95, "max"),
    (6, 2, 0.50, "mean"),
    (7, 20, 0.95, "max"),
]
STATISTICS = {"mean": np.mean, "median": np.median, "max": np.max}


def _data(seed, n):
    rng = np.random.default_rng(100 + seed)
    return rng.normal(10.0, 3.0, size=n)


def _reference(x, statistic, n_resamples, confidence, seed):
    """Independent reference: the documented protocol, written out in full."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    replicates = np.array([statistic(x[row]) for row in idx])
    alpha = 1.0 - confidence
    lo, hi = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear")
    return [float(lo), float(hi)]


def _must_raise_value_error(call, mistake):
    """Fail with the kind of mistake, instead of pytest's generic 'DID NOT RAISE'."""
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


@pytest.mark.parametrize(
    ("seed", "n", "confidence", "name"),
    PROTOCOL_CASES,
    ids=[f"protocol{index}" for index in range(len(PROTOCOL_CASES))],
)
def test_matches_protocol_reference_for_seeded_samples(seed, n, confidence, name):
    x = _data(seed, n)
    statistic = STATISTICS[name]
    # Computed before the call, so a learner who writes into x is reported as an in-place change.
    expected = _reference(x, statistic, 500, confidence, seed)
    np.testing.assert_allclose(
        bootstrap_percentile_ci(x, statistic, n_resamples=500, confidence=confidence, seed=seed),
        expected,
        rtol=1e-12,
        atol=1e-12,
        err_msg="interval differs from the protocol: indices from rng.integers(0, n, (B, n)) and "
        "quantiles with method='linear'",
    )


def test_statistic_is_called_once_per_resample_on_a_one_dimensional_resample():
    x = _data(12, 9)
    shapes = []

    def recording_mean(sample):
        shapes.append(np.shape(sample))
        return float(np.mean(sample))

    bootstrap_percentile_ci(x, recording_mean, n_resamples=40, seed=6)
    assert len(shapes) == 40, (
        "the statistic must be called once per resample: the number of calls is wrong"
    )
    assert all(shape == (9,) for shape in shapes), (
        "each call must receive one 1-D resample of length n, not the whole index matrix"
    )


def test_default_resample_count_and_confidence_follow_the_protocol():
    x = _data(8, 15)
    expected = _reference(x, np.mean, 2000, 0.95, 8)
    np.testing.assert_allclose(
        bootstrap_percentile_ci(x, np.mean, seed=8),
        expected,
        rtol=1e-12,
        atol=1e-12,
        err_msg="default arguments differ from the protocol: check 2000 resamples, 0.95, linear",
    )


def test_higher_confidence_gives_a_nested_wider_interval():
    x = _data(9, 25)
    lo_50, hi_50 = bootstrap_percentile_ci(x, np.mean, n_resamples=800, confidence=0.5, seed=3)
    lo_95, hi_95 = bootstrap_percentile_ci(x, np.mean, n_resamples=800, confidence=0.95, seed=3)
    lo_99, hi_99 = bootstrap_percentile_ci(x, np.mean, n_resamples=800, confidence=0.99, seed=3)
    assert lo_99 <= lo_95 <= lo_50 <= hi_50 <= hi_95 <= hi_99, (
        "with the same seed, a higher confidence level must give a wider, nested interval"
    )


def test_constant_sample_gives_a_point_interval_for_any_statistic():
    for statistic in (np.mean, np.median, np.max):
        lo, hi = bootstrap_percentile_ci([2.5, 2.5, 2.5, 2.5], statistic, seed=4)
        assert (lo, hi) == (2.5, 2.5), "a constant sample has no bootstrap spread: lo must equal hi"


def test_same_seed_reproduces_the_interval_exactly():
    x = _data(10, 11)
    first = bootstrap_percentile_ci(x, np.median, n_resamples=300, seed=21)
    np.testing.assert_allclose(
        bootstrap_percentile_ci(x, np.median, n_resamples=300, seed=21),
        first,
        rtol=0,
        atol=0,
        err_msg="the same seed must reproduce the interval: use default_rng(seed), not globals",
    )


def test_interval_brackets_the_sample_mean_for_a_well_behaved_sample():
    x = _data(11, 50)
    lo, hi = bootstrap_percentile_ci(x, np.mean, seed=12)
    assert lo <= float(np.mean(x)) <= hi, (
        "the percentile interval of the mean should bracket the sample mean for this sample"
    )


INVALID_CASES = [
    ("empty", [], 1, 0.95, "an empty sample has no resamples: raise ValueError"),
    (
        "two_dimensional",
        [[1.0, 2.0], [3.0, 4.0]],
        10,
        0.95,
        "a 2-D sample is not 1-D: raise ValueError",
    ),
    ("nan_value", [1.0, float("nan"), 3.0], 10, 0.95, "a NaN observation must raise ValueError"),
    ("zero_resamples", [1.0, 2.0, 3.0], 0, 0.95, "n_resamples = 0 must raise ValueError"),
    (
        "negative_resamples",
        [1.0, 2.0, 3.0],
        -5,
        0.95,
        "a negative n_resamples must raise ValueError",
    ),
    ("confidence_zero", [1.0, 2.0, 3.0], 10, 0.0, "confidence 0 is not strictly between 0 and 1"),
    ("confidence_one", [1.0, 2.0, 3.0], 10, 1.0, "confidence 1 is not strictly between 0 and 1"),
    ("confidence_above_one", [1.0, 2.0, 3.0], 10, 1.5, "confidence above 1 must raise ValueError"),
]


@pytest.mark.parametrize(
    ("case", "x", "n_resamples", "confidence", "mistake"),
    INVALID_CASES,
    ids=[case[0] for case in INVALID_CASES],
)
def test_invalid_input_raises_value_error(case, x, n_resamples, confidence, mistake):
    _must_raise_value_error(
        lambda: bootstrap_percentile_ci(
            x, np.mean, n_resamples=n_resamples, confidence=confidence, seed=0
        ),
        mistake,
    )


def test_input_array_is_not_mutated():
    x = np.array([4.0, 1.0, 3.0, 2.0])
    before = x.copy()
    bootstrap_percentile_ci(x, np.mean, n_resamples=100, seed=5)
    np.testing.assert_array_equal(
        x,
        before,
        err_msg="the input array was modified in place: resample from a copy instead",
    )


def test_a_single_resample_is_a_valid_request():
    x = [1.0, 2.0, 3.0]
    got = bootstrap_percentile_ci(x, np.mean, n_resamples=1, confidence=0.95, seed=0)
    np.testing.assert_allclose(
        got,
        _reference(x, np.mean, 1, 0.95, 0),
        rtol=1e-12,
        atol=1e-12,
        err_msg="n_resamples = 1 is allowed (at least 1): it must return the interval, not raise",
    )


@pytest.fixture(scope="module")
def iris_sepal_lengths():
    # Measured to 0.1 cm: 150 values with only 35 distinct ones, so many rows repeat.
    return load("iris").X[:, 0]


def test_matches_protocol_on_iris_sepal_lengths_with_repeated_values(iris_sepal_lengths):
    x = iris_sepal_lengths
    expected = _reference(x, np.mean, 600, 0.9, 13)
    np.testing.assert_allclose(
        bootstrap_percentile_ci(x, np.mean, n_resamples=600, confidence=0.9, seed=13),
        expected,
        rtol=1e-12,
        atol=1e-12,
        err_msg="repeated values were resampled as distinct values: every row is one draw, so a "
        "value that appears k times keeps k rows in the pool",
    )


def test_matches_protocol_on_float32_storage_of_iris_sepal_lengths(iris_sepal_lengths):
    stored = iris_sepal_lengths.astype(np.float32)
    expected = _reference(stored.astype(np.float64), np.mean, 600, 0.9, 13)
    np.testing.assert_allclose(
        bootstrap_percentile_ci(stored, np.mean, n_resamples=600, confidence=0.9, seed=13),
        expected,
        rtol=1e-12,
        atol=1e-12,
        err_msg="float32 input is resampled in float32: convert x to float64 before the draws",
    )


def test_a_single_value_sample_gives_a_point_interval():
    try:
        lo, hi = bootstrap_percentile_ci([4.0], np.mean, seed=0)
    except ValueError:
        pytest.fail("a sample of one value is valid: it must return an interval, not raise")
    assert (lo, hi) == (4.0, 4.0), (
        "a sample of one value is valid: every resample repeats it, so the interval is a point"
    )
