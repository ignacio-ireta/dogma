# Public tests: the examples from problem.md plus one edge case (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.

import torch
from submission import group_advantages


def test_binary_group_gets_plus_and_minus_one_with_population_std():
    """Rewards [1, 0, 0, 1]: mean 0.5 and population std 0.5, so advantages are about +-1."""
    out = group_advantages(torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=torch.float64))
    expected = torch.tensor([[1.0, -1.0, -1.0, 1.0]], dtype=torch.float64)
    assert torch.allclose(out, expected, atol=1e-5), (
        "advantage scale is wrong: divide the squared deviations by G (population std), "
        "and subtract the group mean first"
    )


def test_ramp_group_matches_population_std_value():
    """Rewards 1..4: mean 2.5 and population std sqrt(1.25), so advantages are about 1.3416."""
    out = group_advantages(torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64))
    expected = torch.tensor([[-1.341641, -0.447214, 0.447214, 1.341641]], dtype=torch.float64)
    assert torch.allclose(out, expected, atol=1e-5), (
        "advantages do not match the population-std convention: check the mean and divisor G"
    )


def test_single_response_group_is_rejected():
    """A group of one has no spread to normalise by, so the function must raise ValueError."""
    try:
        group_advantages(torch.tensor([[1.0]], dtype=torch.float64))
    except ValueError:
        return
    except Exception as exc:
        raise AssertionError(
            f"a group of one raised {type(exc).__name__}: a group needs at least two responses, "
            "so raise ValueError"
        ) from None
    raise AssertionError(
        "a group of one response was accepted: a group needs at least two responses"
    )
