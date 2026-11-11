from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import cohen_kappa_score
from submission import cohens_kappa


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


@settings(max_examples=100, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(
    st.lists(st.sampled_from(LABELS), min_size=2, max_size=40),
    st.integers(0, 10**6),
)
def test_random_label_sequences_match_the_sklearn_oracle(a, seed):
    rng = np.random.default_rng(seed)
    b = [a[i] if rng.random() < 0.6 else LABELS[int(rng.integers(0, 3))] for i in range(len(a))]
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
    a = [LABELS[i] for i in rng.integers(0, 3, size=20000)]
    b = [LABELS[i] for i in rng.integers(0, 3, size=20000)]
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
