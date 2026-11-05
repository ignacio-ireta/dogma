import numpy as np
import pytest
from submission import sample_discrete


class _Uniforms:
    """A stand-in generator that hands out the given uniforms in order.

    ``random(n)`` returns the next ``n`` values, and ``random()`` returns the next value as a
    scalar.
    """

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)
        self.pos = 0

    def random(self, n=None):
        if n is None:
            value = self.values[self.pos]
            self.pos += 1
            return float(value)
        out = self.values[self.pos : self.pos + n].copy()
        self.pos += n
        return out


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


def test_seed_zero_draws_follow_the_cumulative_intervals():
    draws = sample_discrete([0.2, 0.3, 0.5], 3, np.random.default_rng(0))
    np.testing.assert_array_equal(
        draws,
        [2, 1, 0],
        err_msg="draws differ from the cumulative-interval rule applied to rng.random(3)",
    )


def test_a_draw_on_a_cumulative_boundary_goes_to_the_category_that_starts_there():
    draws = _run(
        lambda: sample_discrete([0.2, 0.3, 0.5], 4, _Uniforms([0.0, 0.2, 0.5, 0.99])),
        "valid inputs must not raise: sample_discrete raised",
    )
    np.testing.assert_array_equal(
        draws,
        [0, 1, 2, 2],
        err_msg="the rule is c[i-1] <= u < c[i]: a draw equal to a cumulative sum starts the next "
        "category",
    )


def test_probabilities_that_do_not_sum_to_one_raise_value_error():
    _raises_value_error(
        lambda: sample_discrete([0.2, 0.3, 0.4], 2, np.random.default_rng(0)),
        "probabilities must sum to 1: raise ValueError when they sum to 0.9",
    )
