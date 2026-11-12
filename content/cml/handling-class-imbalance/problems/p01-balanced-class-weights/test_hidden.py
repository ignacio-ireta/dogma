"""Hidden tests: scikit-learn oracle, the two identities of the rule, label handling and edge cases.

Each failure message names the kind of mistake. No message contains a hidden input or answer.
"""

import functools
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.utils.class_weight import compute_class_weight
from submission import balanced_class_weights

from dogma_data import load_frame, rng

WEIGHT_MISTAKE = (
    "weight differs from the balanced rule: n / (k * count), with k the number of distinct labels"
)


@functools.cache
def _education_years():
    """Years of education of 6000 adults: 16 integer classes, from 8 people to almost 2000."""
    return load_frame("adult")["education.num"].to_numpy()


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def _solve(y):
    """Call the function on input that must be accepted. A crash is a contract mistake."""
    try:
        return balanced_class_weights(y)
    except Exception as exc:
        pytest.fail(
            f"valid input raised {type(exc).__name__}: check the class counts and the formula"
        )


@pytest.mark.parametrize("seed", range(5), ids=[f"case-{s}" for s in range(5)])
def test_matches_scikit_learn_balanced_weights(seed):
    gen = rng(f"balanced-class-weights-p01:sklearn:{seed}")
    n = int(gen.integers(5, 80))
    y = gen.choice(_education_years(), size=n, replace=False)
    classes = np.unique(y)
    expected = compute_class_weight("balanced", classes=classes, y=y)
    got = _solve(y)
    assert set(got) == set(classes.tolist()), "the keys must be exactly the labels that occur"
    for label, weight in zip(classes.tolist(), expected, strict=True):
        assert math.isclose(got[label], weight, rel_tol=1e-12), WEIGHT_MISTAKE


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(st.lists(st.integers(0, 4), min_size=1, max_size=60))
def test_each_class_contributes_n_over_k_to_the_weighted_count(y):
    got = _solve(y)
    n, k = len(y), len(got)
    for label, weight in got.items():
        count = y.count(label)
        assert math.isclose(count * weight, n / k, rel_tol=1e-12), (
            "each class must contribute n / k to the weighted count: check the factor k"
        )
    assert math.isclose(sum(y.count(label) * w for label, w in got.items()), n, rel_tol=1e-12), (
        "the weighted count over all samples must equal n"
    )


@settings(max_examples=100, deadline=None, report_multiple_bugs=False)
@given(st.lists(st.integers(0, 4), min_size=2, max_size=60))
def test_rarer_classes_get_larger_weights(y):
    got = _solve(y)
    for a in got:
        for b in got:
            if y.count(a) < y.count(b):
                assert got[a] > got[b], "a rarer class must get a strictly larger weight"


def test_weights_are_floats_even_for_exact_division():
    got = _solve([0, 1, 0, 1])
    assert all(isinstance(w, float) for w in got.values()), (
        "weights must be floats: check for integer division"
    )
    assert math.isclose(got[0], 1.0, rel_tol=1e-12), WEIGHT_MISTAKE


def test_weights_follow_each_label_not_the_order_of_first_appearance():
    got = _solve(["b", "a", "a", "a", "c"])
    expected = {"a": 5 / 9, "b": 5 / 3, "c": 5 / 3}
    assert set(got) == set(expected), "the keys must be the labels as they appear in y"
    for label, weight in expected.items():
        assert math.isclose(got[label], weight, rel_tol=1e-12), (
            "a weight is attached to the wrong label: match each count to its own label"
        )


def test_single_class_gets_weight_one():
    assert _solve(["x", "x", "x"]) == {"x": 1.0}, "one class: n / (1 * n) = 1.0"


def test_string_labels_are_kept_as_given():
    got = _solve(np.array(["cat", "dog", "cat"]))
    assert set(got) == {"cat", "dog"}, "the dict keys must be the label values, not array positions"


@pytest.mark.parametrize(
    ("y", "mistake"),
    [
        ([], "empty input was accepted: raise ValueError when there are no samples"),
        ([[0, 1], [1, 0]], "two-dimensional input was accepted: raise ValueError for it"),
    ],
    ids=["empty", "two-dimensional"],
)
def test_invalid_shape_raises_value_error(y, mistake):
    _expect_value_error(lambda: balanced_class_weights(y), mistake)


def test_input_is_not_mutated():
    y = np.array([3, 1, 3, 3, 2])
    before = y.copy()
    _solve(y)
    assert np.array_equal(y, before), (
        "y was reordered or changed: count without modifying the input"
    )
