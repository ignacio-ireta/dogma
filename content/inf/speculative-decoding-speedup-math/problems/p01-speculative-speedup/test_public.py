# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import math

from submission import speculative_speedup


def test_mixed_acceptance_gives_closed_form_tokens_and_speedup():
    tokens, speedup = speculative_speedup(0.8, 4, 0.1)
    assert math.isclose(tokens, 3.3616, rel_tol=1e-12), (
        "tokens per round: sum the geometric terms alpha**k for k = 0..gamma"
    )
    assert math.isclose(speedup, 3.3616 / 1.4, rel_tol=1e-12), (
        "speedup: divide by gamma * c + 1, which includes the verification pass"
    )


def test_free_draft_speedup_equals_tokens_per_round():
    tokens, speedup = speculative_speedup(0.5, 2, 0.0)
    assert math.isclose(tokens, 1.75, rel_tol=1e-12), "tokens per round: 1 + 0.5 + 0.25"
    assert math.isclose(speedup, 1.75, rel_tol=1e-12), (
        "with c = 0 the round costs one target pass, so speedup equals tokens"
    )


def test_zero_acceptance_emits_one_token_and_is_slower():
    tokens, speedup = speculative_speedup(0.0, 5, 0.2)
    assert tokens == 1.0, "alpha = 0: only the target's own token is emitted"
    assert math.isclose(speedup, 0.5, rel_tol=1e-12), (
        "alpha = 0: the draft cost is paid and buys nothing"
    )


def test_full_acceptance_emits_gamma_plus_one_tokens():
    tokens, speedup = speculative_speedup(1.0, 3, 0.25)
    assert tokens == 4.0, "alpha = 1: gamma drafts plus the bonus token make gamma + 1"
    assert math.isclose(speedup, 4.0 / 1.75, rel_tol=1e-12), (
        "alpha = 1: round cost is gamma * c + 1 = 1.75"
    )
