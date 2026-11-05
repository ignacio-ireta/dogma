import pytest
from submission import adaboost_round


def _call(y, pred, weights):
    """Call the round; a crash is reported as the broken contract rule, not as a traceback."""
    try:
        return adaboost_round(y, pred, weights)
    except Exception as exc:
        pytest.fail(
            f"a valid round must return (alpha, weights); the call raised "
            f"{type(exc).__name__}: {exc}"
        )


def test_quarter_error_gives_half_log_three_and_the_reweighted_sample():
    alpha, weights = _call([1, 1, 1, 1], [1, 1, 1, -1], [1, 1, 1, 1])
    assert alpha == pytest.approx(0.5493061443340549, abs=1e-12), (
        "alpha = 0.5 * ln((1 - err) / err) with err = 0.25 is 0.5 * ln 3"
    )
    assert weights.tolist() == pytest.approx([1 / 6, 1 / 6, 1 / 6, 0.5], abs=1e-12), (
        "the single mistake must be scaled by e^alpha and the rest by e^-alpha, then normalised"
    )


def test_chance_level_round_is_rejected_with_zero_alpha():
    alpha, weights = _call([1, -1], [1, 1], [1, 1])
    assert alpha == 0.0, "an error of one half means no vote: alpha must be 0"
    assert weights.tolist() == pytest.approx([0.5, 0.5], abs=1e-12), (
        "a rejected round returns the normalised weights unchanged"
    )


def test_perfect_classifier_raises_value_error():
    try:
        adaboost_round([1, -1, 1], [1, -1, 1], [1, 1, 1])
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a perfect classifier must raise ValueError; raised {type(exc).__name__}")
    pytest.fail("a perfect classifier has no finite alpha: it must raise ValueError")
