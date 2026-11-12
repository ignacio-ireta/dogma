# Hidden tests (CONTENT_SPEC §6.4). Tolerance rtol = 1e-12 is declared in meta.yaml.
# The oracles are exact: Fraction for the rational value, and Decimal at 60 digits for
# the near-1 case. A float result within 1e-12 of the exact value is therefore a real
# precision requirement, not a loose check.
import collections
import functools
import itertools
import math
import random
import re
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest
from submission import speculative_speedup

from dogma_data import load_text

RTOL = 1e-12


def call(alpha, gamma, c):
    try:
        result = speculative_speedup(alpha, gamma, c)
    except Exception as error:  # a crash on valid input is a failure, not a test error
        pytest.fail(
            f"valid input raised {type(error).__name__}: check the alpha = 1 and gamma = 0 cases",
            pytrace=False,
        )
    if not (isinstance(result, tuple) and len(result) == 2):
        pytest.fail("return a (tokens_per_round, speedup) tuple", pytrace=False)
    if not all(isinstance(value, float) for value in result):
        pytest.fail("return two Python floats, not ints or arrays", pytrace=False)
    if not all(math.isfinite(value) for value in result):
        pytest.fail("valid inputs must give finite values, not NaN or infinity", pytrace=False)
    return result


def rejects(alpha, gamma, c):
    try:
        speculative_speedup(alpha, gamma, c)
    except ValueError:
        return
    except Exception:
        pytest.fail("invalid input must raise ValueError, not another exception", pytrace=False)
    pytest.fail(
        "invalid input must raise ValueError: check the range and type guards", pytrace=False
    )


def close(actual, expected, mistake):
    if not math.isclose(actual, expected, rel_tol=RTOL, abs_tol=0.0):
        pytest.fail(mistake, pytrace=False)


def exact_tokens(alpha, gamma):
    a = Fraction(alpha)  # the exact binary value of the float, not its decimal rounding
    return sum((a**k for k in range(gamma + 1)), Fraction(0))


@functools.cache
def _acceptance_rates() -> tuple[float, ...]:
    """Real acceptance rates of a draft model against a target model.

    The target is the next-word distribution of Pride and Prejudice and the draft is that of
    Walden. For each word that both books follow up at least twenty times, the acceptance rate
    is the overlap of the two next-word distributions, the sum over words of min(p, q), which
    is the probability that speculative sampling keeps a drafted token.
    """
    follow = []
    for name in ("gutenberg_pride_prejudice", "gutenberg_walden"):
        words = re.findall(r"[a-z]+", load_text(name).lower())
        table: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
        for previous, current in itertools.pairwise(words):
            table[previous][current] += 1
        follow.append(table)
    target, draft = follow
    rates = []
    for word in sorted(set(target) & set(draft)):
        p_total, q_total = sum(target[word].values()), sum(draft[word].values())
        if min(p_total, q_total) >= 20:
            shared = set(target[word]) & set(draft[word])
            rates.append(
                sum(min(target[word][w] / p_total, draft[word][w] / q_total) for w in shared)
            )
    return tuple(rates)


def test_random_inputs_match_exact_rational_geometric_sum():
    rng = random.Random(20260)  # noqa: S311 - seeded oracle inputs, not security
    for _ in range(200):
        alpha = rng.choice(_acceptance_rates())
        gamma = rng.randint(0, 40)
        c = rng.uniform(0.0, 2.0)
        tokens, speedup = call(alpha, gamma, c)
        expected = float(exact_tokens(alpha, gamma))
        close(
            tokens,
            expected,
            "tokens per round differs from the exact geometric sum: "
            "check that the sum runs over k = 0..gamma (gamma + 1 terms)",
        )
        close(
            speedup,
            expected / (gamma * c + 1),
            "speedup must divide tokens per round by gamma * c + 1",
        )


def test_alpha_one_emits_gamma_plus_one_tokens_for_every_draft_length():
    for gamma in (0, 1, 2, 7, 50):
        tokens, speedup = call(1.0, gamma, 0.3)
        assert tokens == gamma + 1, (
            "alpha = 1: the closed form divides by zero, so use the gamma + 1 limit"
        )
        close(
            speedup,
            (gamma + 1) / (gamma * 0.3 + 1),
            "alpha = 1: speedup is (gamma + 1) / (gamma * c + 1)",
        )


def test_alpha_zero_emits_one_token_per_round_for_any_draft_length():
    for gamma in (1, 5, 30):
        tokens, speedup = call(0.0, gamma, 0.2)
        assert tokens == 1.0, "alpha = 0: only the target's own token is emitted"
        close(
            speedup,
            1.0 / (gamma * 0.2 + 1),
            "alpha = 0: speedup is 1 / (gamma * c + 1), below plain decoding",
        )


def test_gamma_zero_is_plain_decoding_for_any_draft_cost_and_alpha():
    for alpha in (0.0, 0.3, 1.0):
        for c in (0.0, 0.5, 7.0):
            tokens, speedup = call(alpha, 0, c)
            assert tokens == 1.0, "gamma = 0: no drafts, so one target token per round"
            assert speedup == 1.0, "gamma = 0: the round is one target pass, speedup is 1"


def test_zero_cost_draft_speedup_equals_tokens_per_round():
    tokens, speedup = call(0.6, 6, 0.0)
    close(speedup, tokens, "with c = 0 the round costs one pass, so speedup equals tokens")


def test_near_one_acceptance_with_long_draft_keeps_relative_precision():
    # alpha = 1 - 1e-12 and gamma = 1000: a**(gamma+1) is 1 - 1e-9 to float precision, so a
    # plain 1 - a**(gamma+1) keeps only about 7 significant digits of the 1e-9 difference.
    alpha = 1.0 - 1e-12
    gamma = 1000
    with localcontext() as ctx:
        ctx.prec = 60
        a = Decimal(alpha)  # exact binary value of the float
        expected = float((1 - a ** (gamma + 1)) / (1 - a))
    tokens, _ = call(alpha, gamma, 0.0)
    close(
        tokens,
        expected,
        "near alpha = 1, rewrite (1 - a**(g+1)) / (1 - a) so the difference is not lost "
        "to rounding (expm1 of the logs)",
    )


def test_large_draft_saturates_at_geometric_limit():
    tokens, _ = call(0.5, 10**6, 0.0)
    close(tokens, 2.0, "long drafts: expected tokens per round tends to 1 / (1 - alpha)")


def test_slow_draft_at_moderate_acceptance_is_slower_than_plain_decoding():
    tokens, speedup = call(0.5, 8, 0.3)
    close(
        tokens,
        (1 - 0.5**9) / 0.5,
        "tokens per round: the closed form is (1 - alpha**(gamma+1)) / (1 - alpha)",
    )
    assert speedup < 1.0, "a draft at 0.3 of target cost at alpha 0.5 must be a net loss"


def test_speedup_does_not_fall_as_acceptance_rises():
    speedups = [call(k / 20, 5, 0.1)[1] for k in range(21)]
    for lower, higher in itertools.pairwise(speedups):
        assert higher >= lower - 1e-15, (
            "speedup must not decrease as the acceptance rate rises (check the sum's terms)"
        )


def test_tokens_per_round_lies_between_one_and_gamma_plus_one():
    rng = random.Random(7)  # noqa: S311 - seeded oracle inputs, not security
    for _ in range(100):
        alpha = rng.choice(_acceptance_rates())
        gamma = rng.randint(0, 60)
        tokens, _ = call(alpha, gamma, rng.uniform(0.0, 1.0))
        assert 1.0 <= tokens <= gamma + 1 + 1e-12, (
            "tokens per round must lie in [1, gamma + 1]: the target always adds one token"
        )


INVALID = [
    (-0.1, 2, 0.1),
    (1.5, 2, 0.1),
    (math.nan, 2, 0.1),
    (math.inf, 2, 0.1),
    (0.5, -1, 0.1),
    (0.5, 2.0, 0.1),
    (0.5, True, 0.1),
    (True, 2, 0.1),
    (0.5, 2, -0.1),
    (0.5, 2, math.nan),
    (0.5, 2, math.inf),
]
INVALID_IDS = [
    "alpha-below-zero",
    "alpha-above-one",
    "alpha-nan",
    "alpha-inf",
    "gamma-negative",
    "gamma-float",
    "gamma-bool",
    "alpha-bool",
    "cost-negative",
    "cost-nan",
    "cost-inf",
]


@pytest.mark.parametrize(("alpha", "gamma", "c"), INVALID, ids=INVALID_IDS)
def test_invalid_alpha_gamma_or_cost_raises_value_error(alpha, gamma, c):
    rejects(alpha, gamma, c)


def test_bool_draft_cost_is_rejected_as_not_a_real_number() -> None:
    # The contract says a bool is not a valid draft cost c, whatever its numeric value.
    rejects(0.5, 2, True)
    rejects(0.5, 2, False)
