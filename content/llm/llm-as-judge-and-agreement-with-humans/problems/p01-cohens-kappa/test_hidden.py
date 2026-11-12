from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr
from sklearn.metrics import cohen_kappa_score
from submission import cohens_kappa

from dogma_data import rng as data_rng
from dogma_data import sample_rows


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


def _raises(expected, call, mistake):
    try:
        call()
    except expected:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


LABELS = ["x", "y", "z"]


def checked(call):
    try:
        value = call()
    except Exception:
        pytest.fail("valid input raised: honor the documented contract", pytrace=False)
    if not isinstance(value, float):
        pytest.fail("return kappa as a plain float", pytrace=False)
    return value


def rejects(call, exc):
    try:
        call()
    except exc:
        return
    except Exception:
        pytest.fail(f"invalid input must raise {exc.__name__}", pytrace=False)
    pytest.fail(f"invalid input must raise {exc.__name__}", pytrace=False)


def is_degenerate(a, b) -> bool:
    # p_e == 1 only when both raters use a single identical label for every item.
    return len(set(a)) == 1 and len(set(b)) == 1 and a[0] == b[0]


def _two_real_raters(seed, n):
    """Two real raters of n iris flowers, with the three species written as x, y and z.

    The first rater is the botanist: the true species. The second is a weak judge that knows
    one measurement and names the species whose average it is closest to. They agree on most
    flowers and disagree where two species overlap, as a human and a model do.
    """
    gen = data_rng(f"cohens-kappa-p01:raters:{seed}:{n}")
    flowers, species = sample_rows("iris", gen, 150)  # every flower, in a drawn order
    known = flowers[:, int(gen.integers(flowers.shape[1]))]
    centres = np.array([known[species == s].mean() for s in range(3)])
    judged = np.abs(known[:n, None] - centres[None, :]).argmin(axis=1)
    return [LABELS[int(s)] for s in species[:n]], [LABELS[int(s)] for s in judged]


@settings(max_examples=100, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(
    st.integers(2, 40),
    st.integers(0, 10**6),
)
def test_random_label_sequences_match_the_sklearn_oracle(n, seed):
    a, b = _two_real_raters(seed, n)
    labels = [*LABELS, "unused"]  # an unused label must not change the result
    if is_degenerate(a, b):
        _raises(
            ValueError,
            lambda: cohens_kappa(a, b, labels),
            "a degenerate pair must raise ValueError",
        )
        return
    got = checked(lambda: cohens_kappa(a, b, labels))
    want = float(cohen_kappa_score(a, b, labels=labels))
    if not math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9):
        pytest.fail(
            "kappa differs from the chance-corrected value: check p_e as the sum of products of "
            "marginals, and the denominator 1 - p_e",
            pytrace=False,
        )


def test_skewed_labels_show_that_raw_agreement_is_not_kappa():
    # 16 of 20 items agree (raw 0.8), but both raters say "yes" 90 percent of the time, so
    # p_e = 0.9 * 0.9 + 0.1 * 0.1 = 0.82 and kappa = (0.8 - 0.82) / (1 - 0.82) = -1/9.
    a = ["yes"] * 18 + ["no"] * 2
    b = ["no", "no"] + ["yes"] * 16 + ["yes", "yes"]
    got = checked(lambda: cohens_kappa(a, b, ["yes", "no"]))
    assert math.isclose(got, -1 / 9, abs_tol=1e-9), (
        "raw agreement was reported instead of the chance-corrected kappa"
    )


def test_kappa_is_symmetric_in_the_two_raters():
    a = ["x", "y", "y", "z", "x", "x", "z"]
    b = ["x", "x", "y", "z", "y", "x", "z"]
    forward = checked(lambda: cohens_kappa(a, b, LABELS))
    backward = checked(lambda: cohens_kappa(b, a, LABELS))
    assert math.isclose(forward, backward, abs_tol=1e-12), (
        "kappa must not depend on which rater is first"
    )


def test_label_order_in_the_label_set_does_not_change_kappa():
    a = ["x", "y", "y", "z", "x"]
    b = ["x", "y", "z", "z", "y"]
    first = checked(lambda: cohens_kappa(a, b, ["x", "y", "z"]))
    second = checked(lambda: cohens_kappa(a, b, ["z", "x", "y"]))
    assert math.isclose(first, second, abs_tol=1e-12), "the order of the label set must not matter"


def test_large_random_inputs_match_the_oracle():
    rng = np.random.default_rng(12)
    a = [LABELS[i] for i in _real_ints(rng, 0, 3, 20000)]
    b = [LABELS[i] for i in _real_ints(rng, 0, 3, 20000)]
    got = checked(lambda: cohens_kappa(a, b, LABELS))
    want = float(cohen_kappa_score(a, b, labels=LABELS))
    assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9), (
        "large-sample kappa differs: check the normalisation by n in both marginals"
    )
    assert abs(got) < 0.05, "independent random raters should have kappa close to zero"


def test_inputs_are_not_mutated():
    a, b = ["x", "y"], ["y", "y"]
    before = (list(a), list(b))
    checked(lambda: cohens_kappa(a, b, ["x", "y"]))
    assert (a, b) == before, "the rater sequences must not be changed"


def test_length_mismatch_and_empty_input_are_rejected():
    rejects(lambda: cohens_kappa(["x"], ["x", "y"], LABELS), ValueError)
    rejects(lambda: cohens_kappa([], [], LABELS), ValueError)


def test_bad_label_set_is_rejected():
    rejects(lambda: cohens_kappa(["x"], ["x"], []), ValueError)
    rejects(lambda: cohens_kappa(["x"], ["x"], ["x", "x"]), ValueError)


def test_label_outside_the_set_is_rejected():
    rejects(lambda: cohens_kappa(["x", "q"], ["x", "x"], LABELS), ValueError)


def test_non_string_labels_raise_type_error():
    rejects(lambda: cohens_kappa([1, 2], [1, 2], [1, 2]), TypeError)  # type: ignore[list-item]


def test_mixed_str_and_non_str_labels_raise_type_error():
    # One non-str rating among str ratings must still raise TypeError, not ValueError.
    rejects(lambda: cohens_kappa(["x", 1], ["x", "x"], LABELS), TypeError)


def test_non_str_label_in_the_label_set_raises_type_error():
    # A non-str entry in the label set is a non-str label, even when the ratings are all str.
    rejects(lambda: cohens_kappa(["x", "y"], ["x", "y"], ["x", "y", 2]), TypeError)  # type: ignore[list-item]


def test_single_item_is_a_valid_input():
    # One item, disagreeing: p_o = 0, p_a(x) = 1, p_b(x) = 0, p_a(y) = 0, p_b(y) = 1,
    # so p_e = 0 and kappa = (0 - 0) / (1 - 0) = 0.
    got = checked(lambda: cohens_kappa(["x"], ["y"], ["x", "y"]))
    assert got == pytest.approx(0.0, abs=1e-12), (
        "a single item is valid input and its kappa is wrong"
    )


def test_label_list_is_not_reordered():
    # The label list is an input too: calling the function must not sort it.
    labels = ["y", "x"]
    checked(lambda: cohens_kappa(["x", "y"], ["x", "x"], labels))
    assert labels == ["y", "x"], "the label list must not be reordered by the call"


def test_one_label_set_reports_undefined_kappa_not_bad_labels():
    # A label set with a single label is valid. When both raters use that label for every item,
    # the chance agreement is 1, so the error must describe kappa itself, not the label set.
    try:
        cohens_kappa(["x", "x"], ["x", "x"], ["x"])
    except ValueError as exc:
        assert "kappa" in str(exc), (
            "a one-label set is valid input; the error for a degenerate kappa must describe kappa"
        )
        return
    pytest.fail("a degenerate kappa must raise ValueError", pytrace=False)
