# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import numpy as np
import pytest
from submission import accept_or_resample


class ScriptedRng:
    """Stand-in generator: random() returns the scripted values in order and counts calls."""

    def __init__(self, *values: float) -> None:
        self._values = list(values)
        self.calls = 0

    def random(self) -> float:
        if not self._values:
            raise AssertionError("drew more uniforms than the draw order allows")
        self.calls += 1
        return self._values.pop(0)


def test_accepted_draft_token_is_returned_with_one_draw():
    rng = ScriptedRng(0.99)
    token = accept_or_resample(np.array([0.6, 0.4, 0.0]), np.array([0.2, 0.3, 0.5]), 1, rng)
    assert token == 1, "ratio 4/3 accepts the draft token, so the draft token is returned"
    assert rng.calls == 1, "an accepted token uses exactly one uniform: no resample draw"


def test_rejected_draft_token_is_replaced_from_the_residual():
    rng = ScriptedRng(0.10, 0.70)
    token = accept_or_resample(np.array([0.6, 0.4, 0.0]), np.array([0.2, 0.3, 0.5]), 2, rng)
    assert token == 0, (
        "rejected draft: sample the residual norm(max(0, p - q)) = [0.8, 0.2, 0] with u2 = 0.7"
    )
    assert rng.calls == 2, "a rejection uses two uniforms: accept, then resample"


def test_zero_draft_probability_for_the_draft_token_raises_value_error():
    with pytest.raises(ValueError):
        accept_or_resample(np.array([0.5, 0.5]), np.array([0.0, 1.0]), 0, ScriptedRng(0.5))
