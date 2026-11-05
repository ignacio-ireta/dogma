# Hidden tests (CONTENT_SPEC §6.4). The oracle sorts the scores and takes the rank from alpha's
# decimal text (repr) with Decimal arithmetic at 60 digits and a ceiling, which is independent of
# the learner's route. Masks are compared exactly. The coverage simulation uses a 4-sigma bound over
# 4000 seeded trials.
import math
from decimal import ROUND_CEILING, Decimal, localcontext

import numpy as np
import pytest
from submission import prediction_sets


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


def random_probs(rng, rows, classes):
    return rng.dirichlet(np.ones(classes), size=rows)


def test_random_inputs_match_sorted_score_oracle():
    rng = np.random.default_rng(77)
    alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9]
    for _ in range(60):
        n = int(rng.integers(1, 26))
        classes = int(rng.integers(2, 7))
        alpha = alphas[int(rng.integers(len(alphas)))]
        cal_probs = random_probs(rng, n, classes)
        cal_labels = rng.integers(0, classes, size=n)
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
    rng = np.random.default_rng(3)
    cal_probs = random_probs(rng, 12, 4)
    cal_labels = rng.integers(0, 4, size=12)
    for alpha in (0.01, 0.2, 0.5, 0.99):
        test = np.eye(4)  # one row per class, probability 1 on that class
        mask = call(cal_probs, cal_labels, test, alpha)
        assert np.all(mask[np.arange(4), np.arange(4)]), (
            "a class with probability 1 has score 0, which is at most any threshold"
        )


def test_sets_are_nested_as_alpha_decreases():
    rng = np.random.default_rng(12)
    cal_probs = random_probs(rng, 30, 5)
    cal_labels = rng.integers(0, 5, size=30)
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
    rng = np.random.default_rng(4)
    cal_probs = random_probs(rng, 10, 3)
    cal_labels = rng.integers(0, 3, size=10)
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
        probs = rng.dirichlet(np.ones(classes), size=n + 1)
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
