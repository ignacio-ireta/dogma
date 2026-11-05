import pytest
from submission import entropy


def _call(p, **kwargs):
    """Call entropy; a crash is reported as the broken contract rule, not as a traceback."""
    try:
        return entropy(p, **kwargs)
    except Exception as exc:
        pytest.fail(
            f"a valid probability vector must return a value; the call raised {type(exc).__name__}"
        )


def test_fair_coin_is_one_bit():
    value = _call([0.5, 0.5])
    assert value == pytest.approx(1.0, abs=1e-12), (
        "fair coin must have entropy 1 bit: the default base is 2"
    )


def test_three_outcome_example_is_one_and_a_half_bits():
    value = _call([0.5, 0.25, 0.25])
    assert value == pytest.approx(1.5, abs=1e-12), (
        "wrong entropy for [0.5, 0.25, 0.25]: average -log2 p over the outcomes"
    )


def test_certain_outcome_has_zero_entropy_because_zero_terms_add_nothing():
    value = _call([1.0, 0.0])
    assert value == pytest.approx(0.0, abs=1e-12), (
        "a certain outcome has zero entropy: skip the zero-probability terms, not return nan"
    )


def test_negative_entry_raises_value_error():
    try:
        entropy([-0.2, 1.2])
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a negative probability must raise ValueError; raised {type(exc).__name__}")
    pytest.fail("a negative probability must raise ValueError, not return a value")
