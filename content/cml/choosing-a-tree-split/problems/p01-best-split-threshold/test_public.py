import pytest
from submission import best_split


def _call(x, y, **kwargs):
    """Call best_split; a crash is reported as the broken contract rule, not as a traceback."""
    try:
        return best_split(x, y, **kwargs)
    except Exception as exc:
        pytest.fail(
            f"a valid split request must return a pair; the call raised {type(exc).__name__}: {exc}"
        )


def test_separable_example_splits_between_the_two_groups():
    threshold, decrease = _call([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1])
    assert threshold == pytest.approx(2.5, abs=1e-12), (
        "the midpoint between the last value of one group and the first of the other is 2.5"
    )
    assert decrease == pytest.approx(0.5, abs=1e-12), (
        "a pure split of a 50/50 node decreases Gini by 0.5: weight the children by their sizes"
    )


def test_tie_goes_to_the_smallest_threshold():
    threshold, decrease = _call([0.0, 1.0, 2.0, 3.0], [0, 1, 1, 0])
    assert threshold == pytest.approx(0.5, abs=1e-12), (
        "thresholds 0.5 and 2.5 tie exactly; the tie must go to the smaller threshold"
    )
    assert decrease == pytest.approx(1.0 / 6.0, abs=1e-12), (
        "the Gini decrease at the chosen threshold should be 1/6"
    )


def test_constant_feature_has_no_valid_split():
    assert _call([5.0, 5.0, 5.0], [0, 1, 0]) == (None, 0.0), (
        "a feature with a single distinct value has no split: return (None, 0.0)"
    )


def test_unknown_criterion_raises_value_error():
    try:
        best_split([0.0, 1.0], [0, 1], criterion="misclassification")
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"an unknown criterion must raise ValueError; raised {type(exc).__name__}")
    pytest.fail("an unknown criterion must raise ValueError, not return a split")
