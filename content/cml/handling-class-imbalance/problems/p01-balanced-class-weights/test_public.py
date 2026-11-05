import math

import pytest
from submission import balanced_class_weights


def _expect_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_two_classes_get_n_over_k_times_count():
    got = balanced_class_weights([0, 0, 0, 1])
    assert set(got) == {0, 1}, "one weight per label that occurs"
    assert math.isclose(got[0], 4 / (2 * 3), rel_tol=1e-12), "weight is n / (k * count)"
    assert math.isclose(got[1], 4 / (2 * 1), rel_tol=1e-12), "weight is n / (k * count)"


def test_three_classes_with_string_labels_keep_their_names():
    got = balanced_class_weights([2, 0, 0, 1, 0, 0, 2, 0, 1])
    expected = {0: 9 / (3 * 5), 1: 9 / (3 * 2), 2: 9 / (3 * 2)}
    assert set(got) == set(expected) and all(
        math.isclose(got[k], expected[k], rel_tol=1e-12) for k in expected
    ), "weights must be n / (k * count) for each label"


def test_empty_input_raises_value_error():
    _expect_value_error(
        lambda: balanced_class_weights([]),
        "empty input was accepted: raise ValueError when there are no samples",
    )
