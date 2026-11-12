# Hidden tests (CONTENT_SPEC §6.4). Failure messages name the kind of mistake and never show
# hidden inputs, shapes or expected values. The oracle is scipy.special (rel_entr, softmax,
# log_softmax); the learner may not import it.
from __future__ import annotations

import copy

import numpy as np
import pytest
from scipy.special import log_softmax, ndtr, rel_entr, softmax
from submission import kl_divergence_from_logits

from dogma_data import rng as data_rng
from dogma_data import sample_rows

ATOL = 1e-10  # the oracle works in float64 from the same logits, so round-off stays near 1e-14
RTOL = 1e-10
LN2 = np.log(2.0)


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


def _reference(lp, lq):
    """KL(softmax(lp) || softmax(lq)) per row, using scipy's relative entropy."""
    return np.sum(rel_entr(softmax(lp, axis=1), softmax(lq, axis=1)), axis=1)


def _reference_log_space(lp, lq):
    """Same KL, but every quantity stays in log space; exact where scipy's softmax underflows."""
    logp, logq = log_softmax(lp, axis=1), log_softmax(lq, axis=1)
    p = np.exp(logp)
    with np.errstate(invalid="ignore"):
        terms = np.where(p > 0, p * (logp - logq), 0.0)
    return np.sum(terms, axis=1)


def _cross_entropy(lp, lq):
    """-sum_j p_j log q_j per row (the mistake of returning cross-entropy instead of KL)."""
    p = softmax(lp, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        terms = np.where(p > 0, -p * log_softmax(lq, axis=1), 0.0)
    return np.sum(terms, axis=1)


def _call(lp, lq):
    try:
        out = kl_divergence_from_logits(lp, lq)
    except Exception as exc:
        pytest.fail(
            f"valid logits were rejected or crashed ({type(exc).__name__}): check the contract"
        )
    if not isinstance(out, np.ndarray) or out.dtype != np.float64:
        pytest.fail("return a float64 NumPy array, one KL value per row")
    if out.shape != (np.shape(lp)[0],):
        pytest.fail("output shape must be (n,): one value per row, not a scalar or a matrix")
    if np.isnan(out).any():
        pytest.fail(
            "nan in the output: a zero probability must contribute 0 (0 log 0 = 0), and the "
            "log-probabilities must come from log-sum-exp"
        )
    return out


def _close(actual, expected):
    return actual.shape == expected.shape and np.allclose(actual, expected, atol=ATOL, rtol=RTOL)


def _assert_kl(out, lp, lq):
    expected = _reference(lp, lq)
    if _close(out, expected):
        return
    if _close(out, _reference(lq, lp)):
        pytest.fail("KL reversed: it computed KL(Q || P) instead of KL(P || Q)")
    if _close(out, _reference(lp, lq) / LN2):
        pytest.fail("the value is in bits: use the natural log, since KL is in nats")
    if _close(out, _cross_entropy(lp, lq)):
        pytest.fail("returned cross-entropy, not KL: subtract the entropy of P")
    if _close(out, expected / lp.shape[1]):
        pytest.fail("averaged over classes: KL sums the terms over classes, it does not average")
    if np.any(np.isposinf(expected) & np.isfinite(out)):
        pytest.fail("p_j > 0 where q_j = 0 must make KL +inf: a positive-p term was dropped")
    pytest.fail("KL differs from the reference: check the roles of P and Q and the zero rules")


def _expect_value_error(lp, lq, mistake):
    try:
        kl_divergence_from_logits(lp, lq)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; the call raised {type(exc).__name__} instead of ValueError")
    pytest.fail(f"{mistake}; the call returned a value instead of raising ValueError")


def _random_logits(seed, n, k, masked_fraction):
    # Real scores standing in for logits: n tumours by k standardised breast_cancer
    # measurements, doubled. They are skewed and correlated across a row. The masked entries
    # are placed by the draw.
    rng = data_rng(f"kl-from-logits-p01:logits:{seed}:{n}:{k}")
    z = 2.0 * sample_rows("breast_cancer", rng, n, k, standardize=True)[0]
    mask = _real_unit(rng, (n, k)) < masked_fraction
    mask[:, 0] = False  # keep one finite logit per row, so the distribution is defined
    z[mask] = -np.inf
    return z


@pytest.mark.parametrize(
    "n, k, masked",
    [(1, 2, 0.0), (4, 3, 0.0), (5, 7, 0.3), (3, 10, 0.5), (8, 2, 0.2)],
    ids=["single-binary", "small", "masked-seven", "half-masked", "many-rows"],
)
def test_matches_scipy_relative_entropy_on_random_batches(n, k, masked):
    lp = _random_logits(100 + n + k, n, k, masked)
    lq = _random_logits(200 + n + k, n, k, masked)
    _assert_kl(_call(lp, lq), lp, lq)


def test_forward_divergence_is_not_the_reverse_divergence():
    lp = np.array([[3.0, 0.0, -1.0]])
    lq = np.array([[0.0, 0.0, 2.0]])
    out = _call(lp, lq)
    assert not np.isclose(_reference(lp, lq)[0], _reference(lq, lp)[0]), "test setup is degenerate"
    _assert_kl(out, lp, lq)


def test_value_is_in_nats_not_bits():
    lp = np.array([[0.0, 1.0, 2.0]])
    lq = np.array([[2.0, 1.0, 0.0]])
    _assert_kl(_call(lp, lq), lp, lq)


def test_cross_entropy_minus_entropy_identity_holds():
    lp = np.array([[0.5, -0.2, 1.0], [2.0, 0.0, -2.0]])
    lq = np.array([[0.1, 0.4, -0.3], [0.0, 0.0, 0.0]])
    p = softmax(lp, axis=1)
    entropy = -np.sum(p * log_softmax(lp, axis=1), axis=1)
    out = _call(lp, lq)
    if not _close(out, _cross_entropy(lp, lq) - entropy):
        pytest.fail("KL must equal cross-entropy minus the entropy of P: check the entropy sign")


def test_large_logits_do_not_overflow():
    lp = np.array([[1000.0, 0.0, -1000.0], [800.0, 799.0, 0.0]])
    lq = np.array([[0.0, 1000.0, 0.0], [700.0, 700.0, 700.0]])
    out = _call(lp, lq)
    # Row 0 has KL about 1000: scipy's softmax underflows q_0 to 0, so the oracle is log-space.
    if not np.allclose(out, _reference_log_space(lp, lq), atol=ATOL, rtol=RTOL):
        pytest.fail("large logits overflow or lose precision: subtract the row maximum first")


def test_underflowing_probability_gives_its_log_ratio_not_nan():
    # P is effectively (1, 0) and Q is effectively (0, 1); exp(-800) underflows to 0 in float64,
    # yet log p and log q stay finite. The exact KL is about 800.
    lp = np.array([[0.0, -800.0]])
    lq = np.array([[-800.0, 0.0]])
    out = _call(lp, lq)
    if not np.isclose(out[0], 800.0, rtol=1e-9):
        pytest.fail("log-probabilities underflow: compute them from log-sum-exp, not log(exp)")


def test_positive_probability_under_zero_q_is_infinite_and_zero_p_adds_nothing():
    lp = np.array([[0.0, -np.inf], [0.0, -np.inf], [0.0, 0.0]])
    lq = np.array([[-np.inf, 0.0], [0.0, -np.inf], [0.0, 0.0]])
    out = _call(lp, lq)
    if not np.isposinf(out[0]):
        pytest.fail("p_j > 0 where q_j = 0 must make the divergence +inf")
    if not np.isfinite(out[1]) or not np.isclose(out[1], 0.0, atol=0.0):
        pytest.fail("a class with p_j = 0 adds nothing, even when q_j = 0: this row has KL 0")
    if not np.isfinite(out[2]):
        pytest.fail("a finite row with no zero probabilities must have a finite KL")


def test_rows_are_computed_independently():
    lp = _random_logits(31, 6, 4, 0.2)
    lq = _random_logits(32, 6, 4, 0.2)
    together = _call(lp, lq)
    for row in range(lp.shape[0]):
        alone = _call(lp[row : row + 1], lq[row : row + 1])
        if not np.allclose(together[row : row + 1], alone, atol=ATOL, rtol=RTOL):
            pytest.fail("rows interact: a reduction runs over the batch axis instead of classes")


def test_integer_logits_are_accepted_and_match_float_logits():
    lp = np.array([[0, 1, 2], [3, 0, 0]])
    lq = np.array([[2, 1, 0], [0, 0, 1]])
    out = _call(lp, lq)
    _assert_kl(out, lp.astype(float), lq.astype(float))


def test_identical_rows_have_zero_divergence_for_many_scales():
    base = np.array([[0.2, -1.0, 3.0, 0.0]])
    for scale in (1e-6, 1.0, 300.0):
        out = _call(base * scale, base * scale)
        if not np.all(np.abs(out) <= 1e-12):
            pytest.fail("identical distributions must have KL 0 at every logit scale")


def test_invalid_inputs_raise_value_error():
    good = np.zeros((2, 3))
    cases = [
        (np.zeros(3), good, "one-dimensional input was accepted"),
        (np.zeros((0, 3)), np.zeros((0, 3)), "empty batch was accepted"),
        (np.zeros((2, 0)), np.zeros((2, 0)), "zero classes were accepted"),
        (good, np.zeros((2, 2)), "mismatched shapes were accepted"),
        (np.array([[0.0, np.nan, 1.0], [0.0, 0.0, 0.0]]), good, "NaN logit was accepted"),
        (np.array([[0.0, np.inf, 1.0], [0.0, 0.0, 0.0]]), good, "+inf logit was accepted"),
        (np.array([[True, False, True], [False, False, False]]), good, "boolean logits accepted"),
        (np.full((2, 3), 1j), good, "complex logits were accepted"),
        (np.full((2, 3), -np.inf), good, "a row with no finite logit was accepted"),
        (good, np.full((2, 3), -np.inf), "a q row with no finite logit was accepted"),
    ]
    for lp, lq, mistake in cases:
        _expect_value_error(lp, lq, mistake)


def test_inputs_are_not_mutated():
    lp = _random_logits(41, 3, 4, 0.2)
    lq = _random_logits(42, 3, 4, 0.2)
    before_p, before_q = copy.deepcopy(lp), copy.deepcopy(lq)
    kl_divergence_from_logits(lp, lq)
    if not np.array_equal(lp, before_p, equal_nan=True) or not np.array_equal(
        lq, before_q, equal_nan=True
    ):
        pytest.fail("the input logits were modified: never write into the caller's arrays")


def test_shapes_that_broadcast_but_differ_still_raise_value_error():
    lp = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    lq = np.array([[0.0, 0.5, 1.0]])
    _expect_value_error(lp, lq, "p_logits and q_logits have different shapes but were accepted")
    _expect_value_error(lq, lp, "q_logits and p_logits have different shapes but were accepted")


def test_single_class_rows_are_valid_and_have_zero_divergence():
    lp = np.array([[0.3], [-4.0]])
    lq = np.array([[-2.0], [1.5]])
    out = _call(lp, lq)
    if not np.allclose(out, 0.0, atol=ATOL, rtol=0.0):
        pytest.fail(
            "a single class has probability 1 in both distributions, so KL is 0: "
            "one-class rows must be accepted"
        )


def test_one_row_without_finite_logits_raises_even_when_other_rows_are_valid():
    good = np.zeros((2, 3))
    bad = np.array([[0.0, 1.0, 2.0], [-np.inf, -np.inf, -np.inf]])
    _expect_value_error(bad, good, "a p row with no finite logit was accepted next to a valid row")
    _expect_value_error(good, bad, "a q row with no finite logit was accepted next to a valid row")
