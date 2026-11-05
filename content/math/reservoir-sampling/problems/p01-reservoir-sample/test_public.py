import numpy as np
import pytest
from submission import reservoir_sample


def _run(call, mistake):
    """Call the learner on valid inputs. An exception becomes a failure that names the mistake."""
    try:
        return call()
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")


def _raises_value_error(call, mistake):
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


def test_seeded_stream_of_ten_keeps_three_items_in_slot_order():
    sample = _run(
        lambda: reservoir_sample(range(10), 3, np.random.default_rng(0)),
        "valid inputs must not raise: reservoir_sample raised",
    )
    assert sample == [9, 6, 7], (
        "the sample differs from Algorithm R with rng.integers(0, i + 1) draws for seed 0"
    )


def test_stream_shorter_than_k_returns_every_item_in_order():
    sample = _run(
        lambda: reservoir_sample(range(3), 5, np.random.default_rng(0)),
        "valid inputs must not raise: reservoir_sample raised",
    )
    assert sample == [0, 1, 2], "a stream shorter than k should be returned whole, in order"


def test_negative_k_raises_value_error():
    _raises_value_error(
        lambda: reservoir_sample(range(5), -1, np.random.default_rng(0)),
        "k must be non-negative: raise ValueError when k is negative",
    )
