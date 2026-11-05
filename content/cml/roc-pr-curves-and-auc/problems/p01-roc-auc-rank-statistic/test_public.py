import math

import pytest
from submission import roc_auc


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_example_counts_correctly_ordered_pairs():
    got = roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert math.isclose(got, 0.75, rel_tol=1e-12), (
        "AUC must be the share of positive-negative pairs ordered correctly (3 of 4 here)"
    )


def test_tied_pair_counts_as_half():
    got = roc_auc([1, 0, 0, 1], [0.9, 0.9, 0.2, 0.6])
    assert math.isclose(got, 0.625, rel_tol=1e-12), (
        "a tied positive-negative pair must count as 0.5, not as 1 or 0"
    )


def test_all_scores_tied_gives_one_half():
    got = roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    assert math.isclose(got, 0.5, rel_tol=1e-12), "when every score ties, the AUC must be 0.5"


def test_single_class_raises_value_error():
    _expect_value_error(
        lambda: roc_auc([1, 1, 1], [0.1, 0.5, 0.9]),
        "single-class input was accepted: raise ValueError when there is no negative sample",
    )
