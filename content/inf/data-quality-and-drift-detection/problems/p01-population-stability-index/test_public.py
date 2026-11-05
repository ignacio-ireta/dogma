import math

import numpy as np
from submission import population_stability_index


def test_identical_samples_have_zero_psi():
    psi = population_stability_index([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], bins=2)
    assert psi == 0.0, (
        "identical samples have identical shares in every bin, so PSI must be exactly 0"
    )


def test_two_bins_with_one_extra_actual_mass_match_the_closed_form():
    psi = population_stability_index([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0, 4.0, 4.0], bins=2)
    assert abs(psi - math.log(2.0) / 6.0) < 1e-12, (
        "expected shares 1/2, 1/2 and actual shares 2/6, 4/6 give (1/6) ln 2"
    )


def test_empty_actual_bin_is_floored_at_eps():
    psi = population_stability_index([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0], bins=2)
    assert math.isfinite(psi), "an empty bin must be floored at eps, not give an infinite term"
    np.testing.assert_allclose(
        psi,
        6.907742156618760,
        rtol=1e-9,
        err_msg="empty actual bin: the zero share must be replaced by eps before the logarithm",
    )


def test_one_bin_is_rejected():
    try:
        population_stability_index([1.0, 2.0], [1.0, 2.0], bins=1)
    except ValueError:
        return
    raise AssertionError("bins = 1 was accepted: bins must be an integer of at least 2")


def test_infinite_eps_is_rejected_instead_of_producing_a_non_finite_index():
    try:
        population_stability_index([1.0, 2.0], [1.0], bins=2, eps=float("inf"))
    except ValueError:
        return
    raise AssertionError("eps must be finite and positive: infinity cannot replace an empty share")
