import pytest
from submission import mutual_information


def _call(x, y):
    """Call the function; a crash is reported as the broken contract rule, not as a traceback."""
    try:
        return mutual_information(x, y)
    except Exception as exc:
        pytest.fail(
            f"valid labels must return a value; the call raised {type(exc).__name__}: {exc}"
        )


def test_perfectly_dependent_binary_labels_give_ln_two():
    value = _call([0, 0, 1, 1], [0, 0, 1, 1])
    assert value == pytest.approx(0.6931471805599453, abs=1e-12), (
        "identical binary labels must give ln 2 nats, the entropy of the label: check the joint "
        "term and the unit"
    )


def test_independent_labels_give_zero():
    value = _call([0, 0, 1, 1], [0, 1, 0, 1])
    assert abs(value) <= 1e-12, (
        "independent labels must give zero mutual information: check the joint counts"
    )


def test_string_labels_are_compared_by_equality():
    value = _call(["a", "a", "b", "b"], ["x", "x", "x", "x"])
    assert abs(value) <= 1e-12, "a constant second label carries no information"


def test_unequal_lengths_raise_value_error():
    try:
        mutual_information([0, 1, 1], [0, 1])
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"unequal lengths must raise ValueError; raised {type(exc).__name__}")
    pytest.fail("unequal lengths must raise ValueError, not return a value")
