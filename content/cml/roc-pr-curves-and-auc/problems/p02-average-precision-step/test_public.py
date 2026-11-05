import math

import pytest
from submission import average_precision


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_example_is_the_step_sum_of_recall_gain_times_precision():
    got = average_precision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert math.isclose(got, 5 / 6, rel_tol=1e-12), (
        "AP must be the sum of (recall gained) x (precision at that threshold)"
    )


def test_one_tied_threshold_gives_the_prevalence():
    got = average_precision([0, 1, 0], [0.0, 0.0, 0.0])
    assert math.isclose(got, 1 / 3, rel_tol=1e-12), (
        "when all scores tie there is one threshold, so AP equals the positive rate"
    )


def test_perfect_ranking_gives_one():
    got = average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert math.isclose(got, 1.0, rel_tol=1e-12), "a perfect ranking must give AP 1.0"


def test_no_positives_raise_value_error():
    _expect_value_error(
        lambda: average_precision([0, 0, 0], [0.1, 0.5, 0.9]),
        "input with no positives was accepted: raise ValueError when there are no positive samples",
    )
