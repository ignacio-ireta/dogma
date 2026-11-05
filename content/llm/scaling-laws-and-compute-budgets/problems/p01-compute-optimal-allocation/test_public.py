# Public tests: the examples from problem.md plus an edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
from submission import compute_optimal_allocation


def test_billion_parameter_model_at_twenty_tokens_per_param():
    assert compute_optimal_allocation(120 * 10**18, 20) == (10**9, 2 * 10**10), (
        "with C = 6 * N * (20 N), the model must have N = 1e9 parameters and D = 2e10 tokens"
    )


def test_real_square_root_is_rounded_down_so_the_budget_is_not_exceeded():
    n, d = compute_optimal_allocation(1000, 20)
    assert (n, d) == (2, 40), "the real answer is about 2.89; round it down to 2"
    assert 6 * n * d <= 1000, "rounding must never exceed the budget: 6 * N * D <= C"


def test_budget_below_one_model_raises_value_error():
    with pytest.raises(ValueError):
        compute_optimal_allocation(100, 20)
