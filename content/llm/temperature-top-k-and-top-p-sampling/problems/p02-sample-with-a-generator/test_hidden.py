# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of mistake
# and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import math

import pytest
import torch
from submission import sample_token as _learner


def _call(*args, **kwargs):
    try:
        return _learner(*args, **kwargs)
    except ValueError:
        pytest.fail(
            "valid arguments were rejected with ValueError: check the argument rules, and that "
            "probs is left unchanged between calls"
        )
    except Exception as exc:
        pytest.fail(f"valid arguments violated the contract: raised {type(exc).__name__}")


def _expect_value_error(*args, **kwargs):
    try:
        _learner(*args, **kwargs)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid argument must raise ValueError, received {type(exc).__name__}")
    pytest.fail("invalid argument was accepted instead of raising ValueError")


def _draws(probs, seed, n):
    gen = torch.Generator().manual_seed(seed)
    return [_call(probs, gen) for _ in range(n)]


def test_empirical_frequencies_match_the_probabilities():
    probs_list = [0.5, 0.3, 0.15, 0.05]
    probs = torch.tensor(probs_list, dtype=torch.float64)
    n = 10000
    counts = [0] * len(probs_list)
    for token in _draws(probs, seed=17, n=n):
        counts[token] += 1
    for k, p in enumerate(probs_list):
        # Tolerance: five standard errors of a binomial proportion, sqrt(p(1-p)/n). With four
        # categories the chance that a correct sampler fails is far below 1e-5, and the seed is
        # fixed, so this only catches a biased draw.
        tol = 5.0 * math.sqrt(p * (1 - p) / n)
        freq = counts[k] / n
        assert abs(freq - p) <= tol, (
            f"token {k} frequency is off: check the cumulative sums and the draw, not the seed"
        )


def test_zero_probability_tokens_are_never_drawn_at_either_end():
    probs = torch.tensor([0.0, 0.4, 0.0, 0.6, 0.0], dtype=torch.float64)
    seen = set(_draws(probs, seed=3, n=3000))
    assert seen <= {1, 3}, "a zero-probability token was drawn: the cdf boundary is wrong"


def test_nearly_all_mass_on_the_last_token_draws_the_last_token():
    probs = torch.tensor([1e-9, 1.0 - 1e-9], dtype=torch.float64)
    draws = _draws(probs, seed=5, n=2000)
    assert draws.count(1) >= 1990, (
        "mass on the last token was not respected: check the cumulative sums"
    )


def test_one_entry_vocabulary_returns_zero():
    assert _call(torch.tensor([1.0], dtype=torch.float64), torch.Generator().manual_seed(0)) == 0, (
        "a one-entry vocabulary leaves a single token to draw"
    )


def test_return_value_is_a_python_int_in_range():
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    gen = torch.Generator().manual_seed(4)
    for _ in range(50):
        token = _call(probs, gen)
        assert type(token) is int, "the return value must be a Python int, not a tensor"
        assert 0 <= token < 4, "the drawn index is outside the vocabulary"


def test_same_seed_reproduces_the_same_sequence():
    probs = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64)
    assert _draws(probs, seed=42, n=100) == _draws(probs, seed=42, n=100), (
        "same seed gave different tokens: the draw must come only from the passed generator"
    )


def test_different_seeds_give_different_sequences():
    probs = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64)
    assert _draws(probs, seed=1, n=100) != _draws(probs, seed=2, n=100), (
        "different seeds gave the same sequence: the generator argument is being ignored"
    )


def test_result_does_not_depend_on_the_global_seed():
    probs = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64)
    torch.manual_seed(1)
    first = _draws(probs, seed=7, n=50)
    torch.manual_seed(999)
    second = _draws(probs, seed=7, n=50)
    assert first == second, "the draw used the global RNG: pass generator= to every random call"


def test_global_random_state_is_left_unchanged():
    probs = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64)
    torch.manual_seed(123)
    before = torch.get_rng_state()
    gen = torch.Generator().manual_seed(8)
    for _ in range(10):
        _call(probs, gen)
    assert torch.equal(torch.get_rng_state(), before), (
        "the global random state changed: use the passed generator only"
    )


def test_input_probabilities_are_not_mutated():
    probs = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64)
    before = probs.clone()
    gen = torch.Generator().manual_seed(9)
    for _ in range(5):
        _call(probs, gen)
    assert torch.equal(probs, before), "probs was written to: build the cdf in a new tensor"


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        torch.tensor([], dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float32),
        torch.tensor([1.5, -0.5], dtype=torch.float64),
        torch.tensor([0.5, float("nan")], dtype=torch.float64),
        torch.tensor([0.5, 0.4], dtype=torch.float64),
    ],
    ids=["two_dimensional", "empty", "float32", "negative", "nan", "sum_not_one"],
)
def test_invalid_probabilities_raise_value_error(bad):
    _expect_value_error(bad, torch.Generator().manual_seed(0))


def test_missing_generator_raises_value_error():
    _expect_value_error(torch.tensor([0.5, 0.5], dtype=torch.float64), None)


def test_integer_seed_instead_of_generator_raises_value_error():
    _expect_value_error(torch.tensor([0.5, 0.5], dtype=torch.float64), 0)


@pytest.mark.parametrize(
    "probs",
    [[0.5, 0.5 + 1.5e-9], [0.5 - 1.5e-9, 0.5]],
    ids=["sum-too-high", "sum-too-low"],
)
def test_sum_off_by_more_than_one_e_minus_nine_is_rejected(probs):
    _expect_value_error(torch.tensor(probs, dtype=torch.float64), torch.Generator().manual_seed(0))
