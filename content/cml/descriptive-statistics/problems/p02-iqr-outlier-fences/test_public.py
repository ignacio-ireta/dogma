import pytest
from submission import iqr_fences


def test_quartiles_of_five_values_give_the_expected_fences():
    lower, upper = iqr_fences([1, 3, 4, 8, 9])
    assert lower == pytest.approx(-4.5, abs=1e-12), "Q1 = 3, Q3 = 8, IQR = 5, so lower = 3 - 7.5"
    assert upper == pytest.approx(15.5, abs=1e-12), "upper = Q3 + 1.5 * IQR = 8 + 7.5"


def test_one_extreme_value_is_flagged_above_the_upper_fence():
    lower, upper = iqr_fences([1, 3, 4, 8, 9, 100])
    assert upper == pytest.approx(17.0, abs=1e-12), (
        "the quartiles fall between order statistics: interpolate, do not take the nearest value"
    )
    assert lower == pytest.approx(-5.0, abs=1e-12)
    assert upper < 100, "the extreme value must lie outside the fences"


def test_k_zero_gives_the_quartiles_themselves():
    assert iqr_fences([1, 3, 4, 8, 9], k=0) == pytest.approx((3.0, 8.0), abs=1e-12)


def test_empty_input_raises_value_error():
    try:
        iqr_fences([])
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(
            f"empty input raised {type(exc).__name__}: raise ValueError when there are no values"
        )
    pytest.fail("empty input was accepted: raise ValueError when there are no quartiles")
