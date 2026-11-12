import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from submission import bias_variance

from dogma_data import rng, sample_rows


def _real_normal(gen, size=None, loc=0.0, scale=1.0):
    """Real numbers where this test used to draw from a normal distribution.

    Standardised breast_cancer measurements (mean 0 and standard deviation 1 over the dataset,
    skewed, with a few values far out), drawn by ``gen``. A vector is one real measurement
    over that many tumours, a matrix is real tumours by real measurements, and any other
    shape is filled from whole rows. ``loc`` and ``scale`` shift and stretch them as before.
    """
    shape = () if size is None else tuple(int(side) for side in np.atleast_1d(size))
    total = int(np.prod(shape)) if shape else 1
    if len(shape) == 1 and 1 <= shape[0] <= 569:
        values = sample_rows("breast_cancer", gen, shape[0], 1, standardize=True)[0][:, 0]
    elif len(shape) == 2 and 1 <= shape[0] <= 569 and 1 <= shape[1] <= 30:
        values = sample_rows("breast_cancer", gen, shape[0], shape[1], standardize=True)[0]
    else:
        rows = max(1, min(569, -(-total // 30)))
        drawn = sample_rows("breast_cancer", gen, rows, standardize=True)[0]
        values = np.resize(drawn.ravel(), total).reshape(shape)
    values = values * scale + loc
    return values if shape else float(values)


def _real_unit(gen, size=None):
    """Real numbers between 0 and 1 where this test used to draw uniform ones.

    Each is the share of a standard normal that lies below a standardised breast_cancer
    measurement, so the values keep the order, the ties and the skew of real measurements.
    """
    return ndtr(_real_normal(gen, size))


def _real_ints(gen, low, high, size=None):
    """Real whole numbers from ``low`` up to ``high - 1``: the band a real measurement falls in.

    The range is cut into ``high - low`` bands of equal normal probability, and each value is
    the band of one standardised measurement. Real measurements are skewed, so the bands are
    not equally full, and neighbouring values tie as real categories and grades do.
    """
    bands = np.floor(_real_unit(gen, size) * (high - low))
    return (low + np.minimum(bands, high - low - 1)).astype(np.int64)


def _real_shares(gen, alpha, size=None):
    """Real probability vectors: positive real numbers (see `_real_unit`) divided by their sum."""
    k = int(np.size(alpha))
    shape = (k,) if size is None else (int(size), k)
    weights = _real_unit(gen, shape) + 1e-3
    return weights / weights.sum(axis=-1, keepdims=True)


def _call(predictions, f_true, noise_var):
    """Call the learner's function and check the output types, naming the mistake if it is off."""
    try:
        result = bias_variance(predictions, f_true, noise_var)
    except ValueError:
        pytest.fail(
            "a valid input raised ValueError: average over the models (axis 0) and check the shapes"
        )
    except Exception as exc:
        pytest.fail(f"a valid input raised {type(exc).__name__}: check the per-point arrays")
    if not (isinstance(result, tuple) and len(result) == 3):
        pytest.fail("the result must be a (bias2, variance, expected_mse) tuple")
    if not all(type(v) is float for v in result):
        pytest.fail("bias2, variance and expected_mse must be Python floats, averaged over points")
    return result


def _require(condition, mistake):
    if not condition:
        pytest.fail(mistake)


def _direct(preds, f, noise_var):
    """Independent formula: bias and variance from E[X] and E[X**2]; expected from raw errors."""
    preds = np.asarray(preds, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    mean = preds.mean(axis=0)
    bias2 = float(np.mean((mean - f) ** 2))
    variance = float(np.mean(np.mean(preds**2, axis=0) - mean**2))
    expected = float(np.mean((preds - f) ** 2) + noise_var)
    return bias2, variance, expected


def _real_predictions(seed, models, points):
    """Predictions of real refitted models, and the real targets they aim at.

    Each model is a least-squares line fitted to its own bootstrap resample of 40 diabetes
    patients (3 standardised features). ``preds[b, i]`` is model ``b`` at held-out patient ``i``
    and ``f[i]`` is that patient's real disease progression, centred and scaled by round numbers.
    """
    gen = rng(f"bias-variance-p01:predictions:{seed}")
    X, y = sample_rows("diabetes", gen, 40 + points, 3, standardize=True)
    y = (y - 150.0) / 75.0
    design = np.column_stack([np.ones(len(X)), X])
    preds = np.empty((models, points))
    for b in range(models):
        rows = gen.integers(0, 40, size=40)
        beta = np.linalg.lstsq(design[rows], y[rows], rcond=None)[0]
        preds[b] = design[40:] @ beta
    return preds, y[40:]


def _simulate(*, seed, degrees, n_models=300, n_train=60, sigma=0.3, m=50):
    """Polynomial fits of increasing degree to fresh training sets from y = sin(2 pi x) + noise."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, m)
    truth = np.sin(2 * np.pi * grid)
    by_degree = {}
    for degree in degrees:
        preds = np.empty((n_models, m))
        for b in range(n_models):
            x = _real_unit(rng, n_train)
            y = np.sin(2 * np.pi * x) + _real_normal(rng, n_train, scale=sigma)
            preds[b] = np.polyval(np.polyfit(x, y, degree), grid)
        by_degree[degree] = preds
    return truth, by_degree


@pytest.mark.parametrize(
    "seed", [0, 1, 2, 3], ids=["seed-zero", "seed-one", "seed-two", "seed-three"]
)
def test_matches_the_direct_formula_on_random_predictions(seed):
    preds, f = _real_predictions(seed, 12, 7)
    noise = 0.4
    exp_bias2, exp_var, exp_expected = _direct(preds, f, noise)
    got_bias2, got_var, got_expected = _call(preds, f, noise)
    _require(
        np.isclose(got_bias2, exp_bias2, rtol=1e-10, atol=1e-12),
        "bias2 differs from the direct formula: square the error of the average model at each"
        " point, then average over the points",
    )
    _require(
        np.isclose(got_var, exp_var, rtol=1e-10, atol=1e-12),
        "variance differs from the direct formula: measure the spread around the average model"
        " (divisor B), then average over the points",
    )
    _require(
        np.isclose(got_expected, exp_expected, rtol=1e-10, atol=1e-12),
        "expected_mse differs from the direct formula: the squared error against the truth plus"
        " noise_var, averaged over the points",
    )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(
    st.integers(min_value=1, max_value=8),
    st.integers(min_value=1, max_value=8),
    st.integers(min_value=0, max_value=10_000),
    st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
)
def test_expected_mse_is_the_sum_of_bias_variance_and_noise(models, points, seed, noise):
    preds, f = _real_predictions(seed, models, points)
    bias2, variance, expected = _call(preds, f, noise)
    _require(
        abs(expected - (bias2 + variance + noise)) <= 1e-9,
        "expected_mse must equal bias2 + variance + noise_var",
    )
    _require(bias2 >= 0.0 and variance >= 0.0, "bias2 and variance cannot be negative")


def test_constant_models_at_the_truth_have_no_error_beyond_noise():
    f = np.array([0.5, -1.0, 2.0])
    preds = np.tile(f, (6, 1))
    bias2, variance, expected = _call(preds, f, 0.7)
    _require(
        (bias2, variance) == (0.0, 0.0) and abs(expected - 0.7) <= 1e-12,
        "models equal to the truth have no bias or variance: expected error is the noise alone",
    )


def test_single_model_has_no_variance_and_bias_is_its_squared_error():
    f = np.array([1.0, 2.0, 3.0])
    pred = np.array([[2.0, 2.0, 1.0]])
    bias2, variance, _ = _call(pred, f, 0.0)
    _require(variance == 0.0, "one model has no spread across models: variance must be zero")
    _require(
        abs(bias2 - (1.0 + 0.0 + 4.0) / 3.0) <= 1e-12,
        "with one model the bias is its mean squared error against the truth",
    )


def test_bias_is_squared_per_point_not_a_signed_average():
    f = np.array([0.0, 0.0])
    preds = np.array([[1.0, -1.0]])
    bias2, _, _ = _call(preds, f, 0.0)
    _require(
        abs(bias2 - 1.0) <= 1e-12,
        "bias must be squared at each test point before averaging: signed errors cancel",
    )


def test_variance_uses_the_model_count_as_divisor():
    preds = np.array([[0.0], [2.0]])
    _, variance, _ = _call(preds, [1.0], 0.0)
    _require(
        abs(variance - 1.0) <= 1e-12,
        "variance over the models uses divisor B (ddof=0), not B minus one",
    )


def test_simulated_polynomial_fits_bias_falls_and_variance_rises_on_a_fixed_seed():
    truth, by_degree = _simulate(seed=2026, degrees=range(1, 10))
    stats = {d: _call(by_degree[d], truth, 0.09) for d in by_degree}
    _require(
        stats[1][0] > stats[3][0] > stats[5][0],
        "squared bias differs from the expected trend in the supplied fixed simulation",
    )
    _require(
        stats[3][1] < stats[5][1] < stats[9][1],
        "model variance differs from the expected trend in the supplied fixed simulation",
    )
    _require(
        stats[9][2] > min(stats[3][2], stats[5][2]),
        "the expected error of the most flexible fit should exceed that of a moderate degree",
    )


def test_scaling_predictions_truth_and_noise_scales_every_output_by_the_square():
    preds, f = _real_predictions(9, 5, 4)
    base = _call(preds, f, 0.3)
    scaled = _call(3.0 * preds, 3.0 * f, 0.3 * 9.0)
    _require(
        np.allclose(np.array(scaled), 9.0 * np.array(base), rtol=1e-10),
        "scaling predictions, truth and noise by c must scale every output by c squared",
    )


def test_permuting_models_and_test_points_leaves_the_outputs_unchanged():
    preds, f = _real_predictions(10, 6, 5)
    base = _call(preds, f, 0.2)
    shuffled = _call(preds[::-1][:, [4, 0, 3, 1, 2]], f[[4, 0, 3, 1, 2]], 0.2)
    _require(
        np.allclose(np.array(shuffled), np.array(base), rtol=1e-12),
        "the outputs must not depend on the order of models or of test points",
    )


INVALID = [
    (
        "one-dimensional-predictions",
        [1.0, 2.0],
        [1.0, 2.0],
        0.0,
        "predictions must be 2-D, one row per model: a 1-D array is not accepted",
    ),
    (
        "truth-length-mismatch",
        [[1.0, 2.0]],
        [1.0, 2.0, 3.0],
        0.0,
        "f_true must have one value per test point: the lengths differ",
    ),
    (
        "truth-two-dimensional",
        [[1.0, 2.0]],
        [[1.0, 2.0]],
        0.0,
        "f_true must be 1-D: a 2-D truth is not accepted",
    ),
    (
        "no-models",
        np.empty((0, 2)),
        [1.0, 2.0],
        0.0,
        "predictions with no model must raise ValueError",
    ),
    (
        "no-test-points",
        np.empty((3, 0)),
        [],
        0.0,
        "predictions with no test point must raise ValueError",
    ),
    (
        "nan-prediction",
        [[1.0, float("nan")]],
        [1.0, 2.0],
        0.0,
        "a NaN prediction must raise ValueError",
    ),
    (
        "infinite-truth",
        [[1.0, 2.0]],
        [1.0, float("inf")],
        0.0,
        "an infinite truth value must raise ValueError",
    ),
    (
        "negative-noise",
        [[1.0, 2.0]],
        [1.0, 2.0],
        -0.1,
        "a negative noise_var must raise ValueError",
    ),
    ("nan-noise", [[1.0, 2.0]], [1.0, 2.0], float("nan"), "a NaN noise_var must raise ValueError"),
]


@pytest.mark.parametrize(
    ("case", "predictions", "f_true", "noise_var", "mistake"),
    INVALID,
    ids=[case[0] for case in INVALID],
)
def test_invalid_input_raises_value_error(case, predictions, f_true, noise_var, mistake):
    try:
        bias_variance(predictions, f_true, noise_var)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_inputs_are_not_mutated():
    preds = np.array([[1.0, 2.0], [0.5, 1.5]])
    f = np.array([1.0, 1.0])
    preds_before, f_before = preds.copy(), f.copy()
    bias_variance(preds, f, 0.1)
    _require(bool(np.array_equal(preds, preds_before)), "predictions were modified in place")
    _require(bool(np.array_equal(f, f_before)), "f_true was modified in place")


@pytest.mark.parametrize("magnitude", [1e-100, 1e100], ids=["small", "large"])
def test_extreme_magnitudes_preserve_representable_squared_errors(magnitude):
    predictions = np.array([[0.0, 2.0], [2.0, 4.0]]) * magnitude
    truth = np.array([0.0, 1.0]) * magnitude
    got = _call(predictions, truth, 0.5 * magnitude**2)
    expected = np.array([2.5, 1.0, 4.0]) * magnitude**2
    _require(
        np.allclose(got, expected, rtol=1e-9, atol=0.0),
        "retain representable squared errors at small and large magnitudes",
    )


def test_integer_and_float32_predictions_return_python_floats():
    for dtype in (np.int64, np.float32):
        got = _call(np.array([[0, 2], [2, 4]], dtype=dtype), np.array([0, 1], dtype=dtype), 0.5)
        _require(
            np.allclose(got, [2.5, 1.0, 4.0], rtol=1e-9, atol=1e-9),
            "convert numeric inputs to floating arithmetic before computing squared errors",
        )


def test_identical_huge_predictions_at_the_truth_have_zero_error():
    _require(
        _call([[1e308], [1e308]], [1e308], 0.0) == (0.0, 0.0, 0.0),
        "identical finite predictions at the truth have zero bias and variance without overflow",
    )


def test_large_common_offset_preserves_small_representable_prediction_spread():
    center = 1e150
    step = 16.0 * np.spacing(center)
    predictions = center + np.array([[-2.0], [-1.0], [1.0], [2.0]]) * step
    got = _call(predictions, [center], 0.0)
    expected = [0.0, 2.5 * step**2, 2.5 * step**2]
    _require(
        np.allclose(got, expected, rtol=1e-9, atol=0.0),
        "remove a common offset before averaging and squaring small prediction deviations",
    )


def test_variance_averages_the_squared_deviation_of_every_model():
    _, variance, _ = bias_variance([[1.0], [2.0], [3.0]], [2.0], 0.0)
    assert variance == pytest.approx(2.0 / 3.0), (
        "variance is the mean squared deviation of every model from the average (divisor B): "
        "the model that sits at the average still counts"
    )


def test_overflowing_spread_is_infinite_not_nan():
    _, variance, _ = _call([[1e308], [-1e308]], [0.0], 0.0)
    _require(
        not np.isnan(variance) and variance == np.inf,
        "a spread beyond the float64 range must come back as positive infinity, "
        "not as NaN and not as a finite number",
    )


def test_overflowing_mean_with_a_zero_prediction_stays_finite():
    # Models predict 1e308, -1e308 and 0. Their mean is 0, so with truth 0 the squared bias is 0.
    bias2, _, _ = _call([[1e308], [-1e308], [0.0]], [0.0], 0.0)
    _require(
        np.isclose(bias2, 0.0, atol=1e-6),
        "the mean prediction is wrong when huge predictions cancel and one prediction is zero",
    )
