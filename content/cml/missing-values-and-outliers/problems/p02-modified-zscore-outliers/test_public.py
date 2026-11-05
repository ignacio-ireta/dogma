import pytest
from submission import robust_zscore_outliers


def _run(x, threshold=3.5):
    try:
        return robust_zscore_outliers(x, threshold)
    except Exception as exc:
        pytest.fail(f"valid outlier request crashed with {type(exc).__name__}: check the MAD rule")


def test_single_large_value_is_flagged_by_the_modified_score():
    flags = _run([1, 2, 3, 4, 100])
    if flags.tolist() != [False, False, False, False, True]:
        pytest.fail(
            "the modified z-score must flag 100 and nothing else: M = 0.6745 * (x - median) / MAD"
        )


def test_zero_mad_flags_only_values_that_differ_from_the_median():
    flags = _run([5, 5, 5, 5, 9])
    if flags.tolist() != [False, False, False, False, True]:
        pytest.fail("with MAD zero, flag exactly the values that differ from the median")


def test_non_positive_threshold_raises_value_error():
    try:
        robust_zscore_outliers([1, 2, 3], threshold=0)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"a zero threshold must raise ValueError (raised {type(exc).__name__} instead)")
    pytest.fail("a zero threshold was accepted: the threshold must be positive")
