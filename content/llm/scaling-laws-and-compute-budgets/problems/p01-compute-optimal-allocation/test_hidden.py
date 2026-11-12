# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of
# mistake and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import functools
import itertools
import random

import pytest
from submission import compute_optimal_allocation as _learner

from dogma_data import load_text

TWO_53 = 2**53


def _call(budget, r):
    try:
        return _learner(budget, r)
    except (TypeError, ValueError):
        raise
    except Exception as exc:
        pytest.fail(f"valid arguments violated the contract: raised {type(exc).__name__}")


def _expect(exc_type, budget, r):
    try:
        _learner(budget, r)
    except exc_type:
        return
    except Exception as exc:
        pytest.fail(f"invalid argument raised {type(exc).__name__}, expected {exc_type.__name__}")
    pytest.fail(f"invalid argument was accepted instead of raising {exc_type.__name__}")


def _oracle(budget, r):
    """Binary search for the largest N with 6 * N * (r * N) <= budget, in exact integers."""
    lo, hi = 0, 1
    while 6 * r * hi * hi <= budget:
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if 6 * r * mid * mid <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo, r * lo


def test_matches_exhaustive_integer_search_on_small_budgets():
    for r in (1, 2, 20):
        for budget in range(1, 3000):
            if budget < 6 * r:
                _expect(ValueError, budget, r)
                continue
            n = 0
            while 6 * r * (n + 1) * (n + 1) <= budget:
                n += 1
            assert _call(budget, r) == (n, r * n), (
                "result is not the largest model that fits: check the rounding of the root"
            )


def test_exact_square_boundary_above_two_to_the_53_is_not_rounded_down():
    n0 = 123_456_789_012
    r = 20
    budget = 6 * r * n0 * n0
    assert budget > TWO_53, "the boundary case must exceed 2**53 to test float precision"
    assert _call(budget, r) == (n0, r * n0), (
        "at the exact boundary C = 6 r N**2 the model must be N exactly: "
        "a float square root loses digits above 2**53"
    )
    assert _call(budget - 1, r) == (n0 - 1, r * (n0 - 1)), (
        "one unit below the boundary must give N - 1: check the floor on the budget"
    )


def test_random_large_budgets_match_binary_search_oracle():
    rng = random.Random(7)  # noqa: S311 -- seeded synthetic budgets, not security
    for _ in range(200):
        budget = rng.randrange(2**60, 10**30)
        r = rng.choice([1, 2, 20, 100])
        assert _call(budget, r) == _oracle(budget, r), (
            "large-budget answer disagrees with the exact search: check the integer root"
        )


@functools.cache
def _corpus_sizes() -> tuple[int, ...]:
    """Sizes, in words, of real training corpora: the first k lines of a novel, for every k
    that gives at least a hundred words. A word stands for a token."""
    sizes = set()
    for name in ("gutenberg_frankenstein", "gutenberg_pride_prejudice", "gutenberg_walden"):
        total = 0
        for line in load_text(name).splitlines():
            total += len(line.split())
            sizes.add(total)
    return tuple(sorted(size for size in sizes if size >= 100))


def test_budget_is_never_exceeded_and_one_more_parameter_would_not_fit():
    rng = random.Random(8)  # noqa: S311 -- seeded synthetic budgets, not security
    for trial in range(200):
        budget = rng.randrange(6 * 3 * 3, 10**25)
        r = rng.choice([3, 20, 50])
        if trial % 2:
            # Every other budget pays for one pass over a real corpus of D words by the
            # model that corpus calls for: N = D // r parameters, C = 6 * N * D.
            tokens = rng.choice(_corpus_sizes())
            budget = 6 * (tokens // r) * tokens
        n, d = _call(budget, r)
        assert 6 * n * d <= budget, "the chosen model exceeds the budget: 6 * N * D must be <= C"
        assert 6 * (n + 1) * (r * (n + 1)) > budget, (
            "N is not maximal: the next integer would still fit the budget"
        )


def test_float_budget_raises_type_error():
    _expect(TypeError, 1e20, 20)


def test_float_ratio_raises_type_error():
    _expect(TypeError, 120 * 10**18, 20.0)


def test_bool_arguments_raise_type_error():
    _expect(TypeError, True, 1)
    _expect(TypeError, 1000, True)


@pytest.mark.parametrize(
    "budget, r",
    [(0, 20), (-5, 20), (1000, 0), (1000, -3)],
    ids=["zero-budget", "negative-budget", "zero-ratio", "negative-ratio"],
)
def test_non_positive_budget_or_ratio_raises_value_error(budget, r):
    _expect(ValueError, budget, r)


def test_budget_below_six_r_raises_value_error_because_no_model_fits():
    _expect(ValueError, 6 * 20 - 1, 20)


def test_results_are_python_ints_and_tokens_equal_ratio_times_parameters():
    n, d = _call(10**22, 50)
    assert type(n) is int and type(d) is int, "N and D must be Python int, not float or numpy"
    assert d == 50 * n, "D must equal r * N exactly"


def test_more_compute_never_gives_a_smaller_model():
    rng = random.Random(9)  # noqa: S311 -- seeded synthetic budgets, not security
    budgets = sorted(rng.randrange(120, 10**24) for _ in range(300))
    sizes = [_call(b, 20)[0] for b in budgets]
    assert all(a <= b for a, b in itertools.pairwise(sizes)), (
        "a larger budget must not give a smaller model: check the monotone rounding"
    )
