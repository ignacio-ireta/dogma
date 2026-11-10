# Hidden tests (CONTENT_SPEC §6.4). The exact checks count draws and compare residual masses on a
# uniform grid with midpoints, so no floating-point tolerance is needed there. The empirical check
# uses a 5-sigma bound: for N draws the standard error of a frequency is at most sqrt(0.25 / N),
# so 5 * sqrt(0.25 / N) makes a false failure with seeded draws very unlikely.
import numpy as np
import pytest
from submission import accept_or_resample


class ScriptedRng:
    """Stand-in generator for exact draw counts: random() returns the scripted values in order."""

    def __init__(self, *values: float) -> None:
        self._values = list(values)
        self.calls = 0

    def random(self) -> float:
        if not self._values:
            raise AssertionError("drew more uniforms than the draw order allows")
        self.calls += 1
        return self._values.pop(0)


def run(p, q, x, u1, u2=0.5):
    rng = ScriptedRng(u1, u2)
    try:
        token = accept_or_resample(np.asarray(p), np.asarray(q), x, rng)
    except AssertionError as error:
        pytest.fail(str(error), pytrace=False)
    except ValueError as error:
        pytest.fail(f"valid input rejected: {error}", pytrace=False)
    return token, rng.calls


def grid(count=1000):
    return (np.arange(count) + 0.5) / count


def test_draw_count_is_one_on_accept_and_two_on_reject_across_the_threshold():
    p = np.array([0.3, 0.7])
    q = np.array([0.8, 0.2])
    threshold = 0.375  # min(1, p[0] / q[0]), chosen off the midpoint grid
    for u1 in grid():
        token, calls = run(p, q, 0, float(u1))
        expected_calls = 1 if u1 < threshold else 2
        assert calls == expected_calls, (
            "draw count wrong: accept on u1 < min(1, p/q) with one draw, else two draws"
        )
        if u1 < threshold:
            assert token == 0, "accepted draft token must be returned unchanged"


def test_rejected_token_follows_the_normalized_residual_not_p_or_q():
    p = np.array([0.1, 0.5, 0.4])
    q = np.array([0.6, 0.2, 0.2])
    # Residual max(0, p - q) = [0, 0.3, 0.2], normalized to [0, 0.6, 0.4].
    counts = np.zeros(3, dtype=int)
    for u2 in grid():
        token, _ = run(p, q, 0, 0.9, float(u2))  # 0.9 > 1/6, so this always rejects
        counts[token] += 1
    assert counts[0] == 0, "the rejected draft token has zero residual mass, so it cannot return"
    assert abs(counts[1] - 600) <= 2, (
        "residual mass is wrong: use max(0, p - q) and normalize it to sum to 1"
    )
    assert abs(counts[2] - 400) <= 2, (
        "residual mass is wrong: use max(0, p - q) and normalize it to sum to 1"
    )


def test_resample_picks_first_index_whose_cumulative_mass_exceeds_u2():
    # Residual is [0.5, 0.25, 0.25, 0] (total 1, exact binary fractions), so the cumulative
    # masses are [0.5, 0.75, 1.0, 1.0]. The rule is the first index with cumulative mass > u2,
    # so equality moves on.
    p = np.array([0.5, 0.25, 0.25, 0.0])
    q = np.array([0.0, 0.0, 0.0, 1.0])
    assert run(p, q, 3, 0.5, 0.49)[0] == 0, "u2 below the first cumulative mass must pick index 0"
    assert run(p, q, 3, 0.5, 0.5)[0] == 1, (
        "u2 equal to the cumulative mass 0.5 must move to index 1"
    )
    assert run(p, q, 3, 0.5, 0.75)[0] == 2, (
        "u2 equal to the cumulative mass 0.75 must move to index 2"
    )


def test_acceptance_comparison_is_strict_at_the_exact_ratio():
    p = np.array([0.25, 0.75])
    q = np.array([0.5, 0.5])  # ratio p[0]/q[0] = 0.5 exactly
    token, calls = run(p, q, 0, 0.5)
    assert calls == 2, "u1 equal to the ratio must reject: the comparison is strict"
    assert token == 1, "after rejection the residual is [0, 0.25], so the token must be 1"
    token, calls = run(p, q, 0, 0.49)
    assert (token, calls) == (0, 1), "u1 below the ratio must accept the draft token"


def test_zero_residual_mass_falls_back_to_target_distribution():
    # p is elementwise at most q (q sums to 1 + 5e-10, within tolerance), so the residual is zero.
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5 + 5e-10])
    token, calls = run(p, q, 1, 0.9999999995, 0.2)
    assert calls == 2, "a rejected token still uses the second uniform"
    assert token == 0, "with zero residual the resample must come from p, so u2 = 0.2 gives 0"
    token, _ = run(p, q, 1, 0.9999999995, 0.7)
    assert token == 1, (
        "zero residual: the resample must come from p, and u2 = 0.7 is past p's cumulative 0.5"
    )


def test_single_token_vocabulary_always_accepts_with_one_draw():
    token, calls = run([1.0], [1.0], 0, 0.999)
    assert (token, calls) == (0, 1), "V = 1: the ratio is 1, so every u1 accepts"


def test_tiny_draft_probability_gives_a_finite_valid_result():
    token, calls = run([0.5, 0.5], [1e-300, 1.0], 0, 0.9)
    assert (token, calls) == (0, 1), "ratio 5e299 must clip to 1 and accept, without overflow"
    token, calls = run([1e-300, 1.0 - 1e-300], [0.5, 0.5], 0, 0.5)
    assert calls == 2 and token == 1, (
        "a tiny target probability must reject and resample the residual, without NaN"
    )


@pytest.mark.parametrize(
    ("p", "q", "x"),
    [
        ([0.5, 0.6], [0.5, 0.5], 0),  # p does not sum to 1
        ([0.5, 0.5], [1.2, -0.2], 0),  # negative entry in q
        ([float("nan"), 0.5], [0.5, 0.5], 0),  # NaN in p
        ([], [], 0),  # empty vocabulary
        ([[0.5, 0.5]], [[0.5, 0.5]], 0),  # not 1-D
        ([0.3, 0.3, 0.4], [0.5, 0.5], 0),  # length mismatch
        ([0.5, 0.5], [0.5, 0.5], 2),  # draft token outside the vocabulary
        ([0.5, 0.5], [0.5, 0.5], -1),  # negative draft token
        ([0.5, 0.5], [0.5, 0.5], True),  # bool is not an index
        ([0.5, 0.5], [0.5, 0.5], 1.0),  # float is not an index
    ],
    ids=[
        "p-not-normalized",
        "q-negative",
        "p-nan",
        "empty",
        "not-one-dimensional",
        "length-mismatch",
        "draft-token-too-large",
        "draft-token-negative",
        "draft-token-bool",
        "draft-token-float",
    ],
)
def test_invalid_probabilities_or_draft_token_raise_value_error(p, q, x):
    try:
        accept_or_resample(
            np.asarray(p, dtype=float), np.asarray(q, dtype=float), x, ScriptedRng(0.1, 0.1)
        )
    except ValueError:
        return
    except Exception:
        pytest.fail("invalid input must raise ValueError, not another exception", pytrace=False)
    pytest.fail(
        "invalid input must raise ValueError: check the validation of p, q and the index",
        pytrace=False,
    )


def test_empirical_output_distribution_equals_target_within_five_sigma():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    q = np.array([0.1, 0.2, 0.6, 0.1])
    n = 50_000
    draft_rng = np.random.default_rng(11)  # draws the draft token from q
    accept_rng = np.random.default_rng(12)  # the step's own draws
    cdf_q = np.cumsum(q)
    counts = np.zeros(p.shape[0], dtype=int)
    for _ in range(n):
        x = min(int(np.searchsorted(cdf_q, draft_rng.random(), side="right")), p.shape[0] - 1)
        counts[accept_or_resample(p, q, x, accept_rng)] += 1
    frequencies = counts / n
    bound = 5.0 * np.sqrt(0.25 / n)
    assert np.max(np.abs(frequencies - p)) <= bound, (
        "output distribution differs from the target: check the residual and the acceptance rule"
    )


def test_inputs_are_not_mutated():
    p = np.array([0.6, 0.4, 0.0])
    q = np.array([0.2, 0.3, 0.5])
    p_before, q_before = p.copy(), q.copy()
    run(p, q, 2, 0.1, 0.7)
    assert np.array_equal(p, p_before) and np.array_equal(q, q_before), (
        "do not normalize or write into the caller's arrays"
    )


def test_returns_a_python_int():
    token = accept_or_resample(
        np.array([0.6, 0.4, 0.0]), np.array([0.2, 0.3, 0.5]), np.int64(1), ScriptedRng(0.99)
    )
    assert type(token) is int, "return a Python int, not a NumPy integer"


def test_two_dimensional_probability_array_is_rejected_even_with_one_entry():
    # p must be 1-D. A single-entry 2-D array is still the wrong shape.
    try:
        accept_or_resample([[1.0]], [[1.0]], 0, ScriptedRng(0.1, 0.1))
    except ValueError:
        return
    except Exception:
        pytest.fail(
            "a 2-D probability array must raise ValueError, not another exception", pytrace=False
        )
    pytest.fail(
        "p must be 1-D: a 2-D array, even with one entry, is invalid and must raise ValueError",
        pytrace=False,
    )


def test_probability_vector_summing_below_one_is_rejected():
    # The sum must be 1 within 1e-9; a vector that sums to 0.5 is a wrong sum.
    try:
        accept_or_resample([0.25, 0.25], [0.5, 0.5], 0, ScriptedRng(0.1, 0.1))
    except ValueError:
        return
    except Exception:
        pytest.fail(
            "a probability vector with a wrong sum must raise ValueError, not another exception",
            pytrace=False,
        )
    pytest.fail(
        "p must sum to 1: a vector summing to 0.5 is invalid and must raise ValueError",
        pytrace=False,
    )
