# Hidden tests (CONTENT_SPEC §6.4). The oracle sorts the scores and takes the rank from alpha's
# decimal text (repr) with Decimal arithmetic at 60 digits and a ceiling, which is independent of
# the learner's route. Masks are compared exactly. The coverage simulation uses a 4-sigma bound over
# 4000 seeded trials.
import math
from decimal import ROUND_CEILING, Decimal, localcontext

import numpy as np
import pytest
from scipy.special import ndtr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from submission import prediction_sets

from dogma_data import load, rng, sample_rows, synth
from dogma_data import rng as data_rng


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


def exact_rank(n, alpha):
    with localcontext() as context:
        context.prec = 60
        value = (n + 1) * (1 - Decimal(repr(float(alpha))))
        return int(value.to_integral_value(rounding=ROUND_CEILING))


def oracle(cal_probs, cal_labels, test_probs, alpha):
    n = len(cal_labels)
    scores = sorted(1.0 - float(cal_probs[i][cal_labels[i]]) for i in range(n))
    k = exact_rank(n, alpha)
    q = math.inf if k > n else scores[k - 1]
    return np.array([[(1.0 - float(p)) <= q for p in row] for row in test_probs], dtype=bool)


def call(cal_probs, cal_labels, test_probs, alpha):
    try:
        mask = prediction_sets(
            np.asarray(cal_probs), np.asarray(cal_labels), np.asarray(test_probs), alpha
        )
    except Exception as error:  # a crash on valid input is a failure, not a test error
        pytest.fail(
            f"valid input raised {type(error).__name__}: check the score and the rank",
            pytrace=False,
        )
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        pytest.fail("return a boolean NumPy array", pytrace=False)
    if mask.shape != (len(test_probs), np.asarray(cal_probs).shape[1]):
        pytest.fail("return a mask shaped (test rows, K)", pytrace=False)
    return mask


def digit_probs(gen, rows, classes):
    """Real class probabilities and the real labels for digits images of classes 0..classes-1.

    The classifier is nearest centroid on eight randomly chosen pixels: the probabilities are
    the softmax of minus the squared distance to each class's mean image, divided by 20. Eight
    pixels are too few to be sure, so the true class often has a middling probability and is
    sometimes not the top one.
    """
    X, label = sample_rows("digits", gen, 600)
    pixels = gen.choice(X.shape[1], size=8, replace=False)
    X, label = X[:, pixels], label.astype(np.int64)
    centroids = np.stack([X[label == c].mean(axis=0) for c in range(classes)])
    keep = np.flatnonzero(label < classes)[:rows]
    logits = -((X[keep, None, :] - centroids[None, :, :]) ** 2).sum(axis=2) / 20.0
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True), label[keep]


def random_probs(gen, rows, classes):
    return digit_probs(gen, rows, classes)[0]


def test_random_inputs_match_sorted_score_oracle():
    rng = data_rng("conformal-sets-p02:oracle")
    alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9]
    for _ in range(60):
        n = int(rng.integers(1, 26))
        classes = int(rng.integers(2, 7))
        alpha = alphas[int(rng.integers(len(alphas)))]
        cal_probs, cal_labels = digit_probs(rng, n, classes)
        test_probs = random_probs(rng, 6, classes)
        mask = call(cal_probs, cal_labels, test_probs, alpha)
        want = oracle(cal_probs, cal_labels, test_probs, alpha)
        np.testing.assert_array_equal(
            mask,
            want,
            err_msg="set differs from the sorted-score oracle: check the score 1 - p(true class), "
            "the 1-indexed rank k, and the comparison direction",
        )


def test_rank_is_read_from_the_decimal_alpha_not_a_float_product():
    # n = 9 and alpha = 0.7: rank 3, so q is the 3rd smallest score, 0.3. The row has a class with
    # score 0.35, which must be excluded. The float product 10 * (1 - 0.7) gives rank 4, q = 0.4,
    # and would include it.
    true_probs = np.linspace(0.9, 0.1, 9)  # scores 0.1 .. 0.9
    cal_probs = np.column_stack([true_probs, 1.0 - true_probs])
    cal_labels = np.zeros(9, dtype=int)
    mask = call(cal_probs, cal_labels, [[0.65, 0.35]], 0.7)
    np.testing.assert_array_equal(
        mask,
        [[False, False]],
        err_msg="rank for n = 9 and alpha = 0.7 is 3: the threshold is the 3rd smallest score",
    )
    # alpha = 0.3 is 3/10 as written: rank 7, q = 0.7. The float 0.3 is not exactly 3/10, and its
    # binary value gives rank 8, which would include the class with score 0.75.
    mask = call(cal_probs, cal_labels, [[0.25, 0.75]], 0.3)
    np.testing.assert_array_equal(
        mask,
        [[False, True]],
        err_msg="alpha = 0.3 is read as written: rank 7, threshold 0.7",
    )


def test_rank_above_calibration_size_includes_every_class():
    cal_probs = np.array([[0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]])
    mask = call(cal_probs, [0, 0, 0, 0], [[0.99, 0.01], [0.01, 0.99]], 0.1)
    assert mask.all(), (
        "k = ceil(5 * 0.9) = 5 > n = 4: the threshold is infinite, so all classes are in"
    )


def test_confident_prediction_always_keeps_its_class():
    cal_probs, cal_labels = digit_probs(data_rng("conformal-sets-p02:certain-class"), 12, 4)
    for alpha in (0.01, 0.2, 0.5, 0.99):
        test = np.eye(4)  # one row per class, probability 1 on that class
        mask = call(cal_probs, cal_labels, test, alpha)
        assert np.all(mask[np.arange(4), np.arange(4)]), (
            "a class with probability 1 has score 0, which is at most any threshold"
        )


def test_sets_are_nested_as_alpha_decreases():
    rng = data_rng("conformal-sets-p02:nested")
    cal_probs, cal_labels = digit_probs(rng, 30, 5)
    test = random_probs(rng, 20, 5)
    previous = None
    for alpha in (0.9, 0.5, 0.3, 0.1, 0.05):
        mask = call(cal_probs, cal_labels, test, alpha)
        if previous is not None:
            assert np.all(mask >= previous), (
                "smaller alpha must give a superset: check the rank direction "
                "(ceil of (n + 1)(1 - alpha))"
            )
        previous = mask


def test_ties_at_the_threshold_are_included():
    cal_probs = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    cal_labels = np.array([0, 0, 0])
    # every score is 0.5; alpha = 0.5 gives k = ceil(4 * 0.5) = 2, so q = 0.5
    mask = call(cal_probs, cal_labels, [[0.5, 0.5]], 0.5)
    assert mask.all(), "a score equal to the threshold is inside the set"


@pytest.mark.parametrize(
    ("cal_probs", "cal_labels", "test_probs", "alpha"),
    [
        ([[0.5, 0.4]], [0], [[0.5, 0.5]], 0.2),  # calibration row does not sum to 1
        ([[1.2, -0.2]], [0], [[0.5, 0.5]], 0.2),  # negative entry
        ([[0.5, 0.5]], [0.0], [[0.5, 0.5]], 0.2),  # float labels
        ([[0.5, 0.5]], [True], [[0.5, 0.5]], 0.2),  # bool labels
        ([[0.5, 0.5]], [2], [[0.5, 0.5]], 0.2),  # label outside [0, K)
        ([[0.5, 0.5]], [-1], [[0.5, 0.5]], 0.2),  # negative label
        ([[0.5, 0.5]], [0], [[0.4, 0.3, 0.3]], 0.2),  # class-count mismatch
        ([[0.5, 0.5]], [0], [[0.5, 0.5]], True),  # bool alpha
        ([[0.5, 0.5]], [0], [[0.5, 0.5]], 1.0),  # alpha outside (0, 1)
        ([], [], [[0.5, 0.5]], 0.2),  # empty calibration
    ],
    ids=[
        "cal-row-not-normalized",
        "negative-probability",
        "float-labels",
        "bool-labels",
        "label-too-large",
        "label-negative",
        "class-count-mismatch",
        "alpha-bool",
        "alpha-one",
        "empty-calibration",
    ],
)
def test_invalid_probabilities_labels_or_alpha_raise_value_error(
    cal_probs, cal_labels, test_probs, alpha
):
    try:
        prediction_sets(
            np.asarray(cal_probs, dtype=float).reshape(-1, 2) if cal_probs else np.zeros((0, 2)),
            np.asarray(cal_labels),
            np.asarray(test_probs, dtype=float),
            alpha,
        )
    except ValueError:
        return
    except Exception:
        pytest.fail("invalid input must raise ValueError, not another exception", pytrace=False)
    pytest.fail(
        "invalid input must raise ValueError: check the probability, label and alpha guards",
        pytrace=False,
    )


def test_zero_test_rows_returns_empty_mask_with_k_columns():
    cal_probs = np.array([[0.6, 0.4], [0.3, 0.7]])
    mask = call(cal_probs, [0, 1], np.zeros((0, 2)), 0.3)
    assert mask.shape == (0, 2), "no test rows: the mask keeps the class columns"


def test_inputs_are_not_mutated():
    rng = data_rng("conformal-sets-p02:not-mutated")
    cal_probs, cal_labels = digit_probs(rng, 10, 3)
    test_probs = random_probs(rng, 5, 3)
    before = cal_probs.copy(), cal_labels.copy(), test_probs.copy()
    call(cal_probs, cal_labels, test_probs, 0.2)
    assert all(
        np.array_equal(a, b)
        for a, b in zip((cal_probs, cal_labels, test_probs), before, strict=True)
    ), "do not sort, renormalize or write into the caller's arrays"


def test_empirical_coverage_meets_one_minus_alpha_within_a_stated_bound():
    # Perfectly calibrated synthetic classifier: each row of probabilities is drawn from a
    # Dirichlet, and the label is then drawn from that row. The pairs are i.i.d., so they are
    # exchangeable. n = 9 calibration rows, one test row per trial, K = 3, alpha = 0.15.
    # The rank is k = 9, so coverage is k / (n + 1) = 0.9 for continuous scores. The tolerance
    # 4 * sqrt(alpha (1 - alpha) / T) with T = 4000 is about 0.023.
    rng = np.random.default_rng(99)
    alpha, n, classes, trials = 0.15, 9, 3, 4000
    covered = 0
    for _ in range(trials):
        probs = _real_shares(rng, np.ones(classes), n + 1)
        labels = np.array([rng.choice(classes, p=row) for row in probs])
        mask = call(probs[:n], labels[:n], probs[n : n + 1], alpha)
        covered += int(mask[0, labels[n]])
    coverage = covered / trials
    tolerance = 4.0 * math.sqrt(alpha * (1 - alpha) / trials)
    assert coverage >= (1 - alpha) - tolerance, (
        "coverage below 1 - alpha: check the rank (ceil, not floor) and that n + 1 is used"
    )
    assert coverage <= (1 - alpha) + 1.0 / (n + 1) + tolerance, (
        "coverage above the finite-sample ceiling: the rank is too large"
    )


def _must_reject(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as error:
        pytest.fail(f"{mistake}; raised {type(error).__name__} instead", pytrace=False)
    pytest.fail(mistake, pytrace=False)


def test_complex_valued_probabilities_are_rejected():
    cal_probs = np.array([[0.5 + 0j, 0.5 + 0j], [0.2 + 0j, 0.8 + 0j]])
    _must_reject(
        lambda: prediction_sets(cal_probs, [0, 1], [[0.5, 0.5]], 0.2),
        "complex-valued calibration probabilities must raise ValueError: probabilities are real",
    )


def test_label_array_with_wrong_shape_is_rejected():
    cal_probs = np.array([[0.6, 0.4], [0.7, 0.3], [0.2, 0.8], [0.5, 0.5]])
    _must_reject(
        lambda: prediction_sets(cal_probs, np.array([[0], [1], [0], [1]]), [[0.5, 0.5]], 0.2),
        "cal_labels must be one-dimensional with one label per calibration row",
    )


def test_single_class_probabilities_are_rejected():
    cal_probs = np.ones((4, 1))
    _must_reject(
        lambda: prediction_sets(cal_probs, [0, 0, 0, 0], [[1.0]], 0.2),
        "a table with fewer than two classes must raise ValueError: K must be 2 or more",
    )


def test_empty_calibration_set_is_rejected():
    _must_reject(
        lambda: prediction_sets(np.zeros((0, 3)), np.zeros(0, dtype=int), [[0.2, 0.3, 0.5]], 0.2),
        "an empty calibration set must raise ValueError: calibration needs at least one row",
    )


def test_none_alpha_is_rejected():
    _must_reject(
        lambda: prediction_sets([[0.5, 0.5]], [0], [[0.5, 0.5]], None),
        "alpha must be a Python float; a missing alpha must raise ValueError",
    )


def test_negative_label_mixed_with_valid_labels_is_rejected():
    cal_probs = np.array([[0.6, 0.2, 0.2], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6], [0.5, 0.3, 0.2]])
    _must_reject(
        lambda: prediction_sets(cal_probs, [-1, 0, 1, 0], [[0.5, 0.3, 0.2]], 0.2),
        "a negative label anywhere in cal_labels must raise ValueError: labels lie in [0, K)",
    )


def test_out_of_range_label_mixed_with_valid_labels_is_rejected():
    cal_probs = np.array([[0.6, 0.2, 0.2], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6], [0.5, 0.3, 0.2]])
    _must_reject(
        lambda: prediction_sets(cal_probs, [0, 1, 3, 0], [[0.5, 0.3, 0.2]], 0.2),
        "a label equal to K anywhere in cal_labels must raise ValueError: labels lie in [0, K)",
    )


def test_one_unnormalized_calibration_row_among_valid_rows_is_rejected():
    cal_probs = np.array([[0.6, 0.4], [0.7, 0.3], [0.25, 0.25]])
    _must_reject(
        lambda: prediction_sets(cal_probs, [0, 0, 1], [[0.5, 0.5]], 0.2),
        "every calibration row must sum to 1, not only some of them",
    )


PROBLEM = "safe/split-conformal-prediction/p02-conformal-prediction-sets"


def _oracle_masks(cal_probs, cal_labels, test_probs, alpha):
    return oracle(np.asarray(cal_probs), np.asarray(cal_labels), np.asarray(test_probs), alpha)


@pytest.mark.parametrize(("n", "alpha"), [(9, 0.15), (30, 0.1)], ids=["n-9", "n-30"])
def test_coverage_on_over_confident_probabilities_meets_the_finite_sample_bound(n, alpha):
    gen = rng(f"{PROBLEM}:peaked-coverage:{n}:0")
    classes, trials = 10, 3000
    labels = _real_ints(gen, 0, classes, (trials, n + 1))
    probs = synth.peaked_probs(gen, labels.reshape(-1), classes).reshape(trials, n + 1, classes)
    covered = 0
    for t in range(trials):
        mask = call(probs[t, :n], labels[t, :n], probs[t, n : n + 1], alpha)
        covered += int(mask[0, labels[t, n]])
    coverage = covered / trials
    tolerance = 4.0 * math.sqrt(alpha * (1 - alpha) / trials)
    assert coverage >= (1 - alpha) - tolerance, (
        "coverage below 1 - alpha on over-confident scores: the rank must use n + 1, "
        "k = ceil((n + 1)(1 - alpha)), not n"
    )
    assert coverage <= (1 - alpha) + 1.0 / (n + 1) + tolerance, (
        "coverage above the finite-sample ceiling on over-confident scores: the rank is too large"
    )


def test_sets_from_a_real_classifier_match_the_sorted_score_oracle():
    data = load("breast_cancer")
    x = StandardScaler().fit_transform(data.X)
    y = np.asarray(data.y).astype(int)
    gen = rng(f"{PROBLEM}:breast-cancer:0")
    order = gen.permutation(len(y))
    train, cal, test = order[:200], order[200:260], order[260:]
    model = LogisticRegression(max_iter=500).fit(x[train], y[train])
    cal_probs = model.predict_proba(x[cal])
    test_probs = model.predict_proba(x[test])
    for alpha in (0.05, 0.1, 0.3):
        mask = call(cal_probs, y[cal], test_probs, alpha)
        np.testing.assert_array_equal(
            mask,
            _oracle_masks(cal_probs, y[cal], test_probs, alpha),
            err_msg="set differs from the sorted-score oracle on real classifier probabilities: "
            "check the score 1 - p(true class) and the rank with n + 1",
        )


def test_scores_are_compared_in_float64_so_nearby_probabilities_stay_apart():
    # Over-confident classifiers report probabilities a billionth apart. Calibration scores are
    # 0.3 + 1e-9 (n = 9, alpha = 0.7, so q is that common value). A test score 0.3 + 2e-9 is above
    # q and must be out; 0.3 + 0.5e-9 is below q and must be in. Float32 scores round both
    # test values to the same number, so the first would wrongly enter.
    cal_probs = np.tile([0.7 - 1e-9, 0.3 + 1e-9], (9, 1))
    mask = call(
        cal_probs,
        np.zeros(9, dtype=int),
        [[0.7 - 2e-9, 0.3 + 2e-9], [0.7 - 0.5e-9, 0.3 + 0.5e-9]],
        0.7,
    )
    np.testing.assert_array_equal(
        mask,
        [[False, False], [True, False]],
        err_msg="scores must be compared in float64: a score 1e-9 above the threshold is outside",
    )


def test_alpha_at_the_open_lower_end_is_rejected():
    cal_probs = [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5], [0.5, 0.4, 0.1]]
    _must_reject(
        lambda: prediction_sets(cal_probs, [0, 1, 2, 0], [[0.7, 0.2, 0.1]], 0.0),
        "alpha must lie strictly between 0 and 1: the lower end is excluded, "
        "so it must raise ValueError",
    )


def test_list_inputs_give_the_same_result_as_arrays():
    cal = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.5, 0.3, 0.2]])
    labels = np.array([0, 1, 2, 0])
    test = np.array([[0.7, 0.2, 0.1], [0.3, 0.3, 0.4]])
    try:
        from_lists = prediction_sets(cal.tolist(), labels, test.tolist(), 0.4)
    except Exception:
        pytest.fail(
            "a valid call with Python lists failed: the inputs are array-like, so convert them "
            "with np.asarray before using array methods",
            pytrace=False,
        )
    from_arrays = prediction_sets(cal, labels, test, 0.4)
    assert np.array_equal(from_lists, from_arrays), (
        "lists and arrays of the same numbers gave different results"
    )
