import itertools

import numpy as np
import pytest
from submission import permutation_test_mean_diff

# Small samples whose every relabelling can be enumerated. The last one has exact ties in the mean
# difference once rounded in floating point, which is where a naive comparison goes wrong.
EXACT_CASES = [
    ([1.2, 2.5, 3.1], [2.0, 3.3, 4.8]),
    ([4.1, 5.0, 6.2, 5.5], [3.0, 3.8, 4.4, 2.9]),
    ([10.0, 11.0, 12.0], [10.0, 13.0, 14.0, 15.0]),
    ([1.0, 2.0, 3.0], [7.0, 8.0, 9.0]),
    ([0.3, 0.9, 1.4, 2.2], [0.1, 0.4, 0.6, 1.1, 1.5]),
]
TIE_CASE = ([0.1, 0.5, 0.1], [0.3, 0.3, 0.5, 0.3])
MC_TOLERANCE = 0.02
N_PERM = 10000


def _exact_p(a, b):
    """Exact two-sided p-value: enumerate every split of the pooled sample into the two sizes."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pooled = np.concatenate([a, b])
    n_a = a.size
    observed = abs(a.mean() - b.mean())
    hits = 0
    total = 0
    for chosen in itertools.combinations(range(pooled.size), n_a):
        mask = np.zeros(pooled.size, dtype=bool)
        mask[list(chosen)] = True
        difference = abs(pooled[mask].mean() - pooled[~mask].mean())
        total += 1
        if difference >= observed * (1.0 - 1e-9):
            hits += 1
    return hits / total


def _must_raise_value_error(call, mistake):
    """Fail with the kind of mistake, instead of pytest's generic 'DID NOT RAISE'."""
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


@pytest.mark.parametrize(
    ("a", "b"),
    EXACT_CASES,
    ids=[f"exact{index}" for index in range(len(EXACT_CASES))],
)
def test_p_value_is_close_to_exact_enumeration(a, b):
    p = permutation_test_mean_diff(a, b, n_permutations=N_PERM, seed=0)
    assert abs(p - _exact_p(a, b)) <= MC_TOLERANCE, (
        "p-value is far from the exact enumeration: check the statistic (mean difference), "
        "the pooling and the two-sided comparison"
    )


def test_exact_float_ties_are_counted_as_at_least_as_extreme():
    a, b = TIE_CASE
    p = permutation_test_mean_diff(a, b, n_permutations=N_PERM, seed=0)
    assert abs(p - _exact_p(a, b)) <= MC_TOLERANCE, (
        "relabellings that tie the observed difference are counted too rarely: compare with a "
        "relative tolerance, because float rounding splits exact ties"
    )


def test_a_near_tie_below_the_observed_difference_is_not_counted():
    # Two relabellings sit about 4e-4 (relative) below the observed difference. They are not ties,
    # so the tolerance must stay near 1e-9. A looser tolerance, such as 1e-3, would count them and
    # move the exact p-value from 0.30 to 0.40.
    a, b = [1255.0, 1438.0, 1722.0], [3913.0, 3949.0, 1256.0]
    p = permutation_test_mean_diff(a, b, n_permutations=N_PERM, seed=0)
    assert abs(p - _exact_p(a, b)) <= MC_TOLERANCE, (
        "a relabelling just below the observed difference was counted as a tie: keep the relative "
        "tolerance near 1e-9, not 1e-3"
    )


def test_p_value_is_the_same_whichever_group_comes_first():
    for a, b in EXACT_CASES[:2]:
        for first, second in ((a, b), (b, a)):
            p = permutation_test_mean_diff(first, second, n_permutations=N_PERM, seed=1)
            assert abs(p - _exact_p(a, b)) <= MC_TOLERANCE, (
                "a two-sided test must not depend on group order: use the absolute difference"
            )


def test_identical_groups_have_p_equal_to_one_exactly():
    assert permutation_test_mean_diff([2.0, 2.0, 2.0], [2.0, 2.0, 2.0], seed=3) == 1.0, (
        "identical groups: every relabelling ties the observed difference, so p must be exactly 1"
    )


def test_p_value_has_the_plus_one_form_and_is_never_zero():
    # Groups of 8 and 8 with no overlap: only two of the 12870 splits are this extreme, so the
    # observed labelling is almost never drawn. Without the +1 the p-value would be exactly zero.
    a = np.arange(8, dtype=np.float64) + 100.0
    b = np.arange(8, dtype=np.float64)
    n_perm = 99
    p = permutation_test_mean_diff(a, b, n_permutations=n_perm, seed=0)
    scaled = p * (n_perm + 1)
    assert abs(scaled - round(scaled)) < 1e-9 and round(scaled) >= 1, (
        "p must equal (count + 1) / (n_permutations + 1) for some count >= 0: "
        "check the +1 in the numerator and the denominator"
    )


def test_same_seed_gives_the_same_p_value():
    a, b = EXACT_CASES[1]
    first = permutation_test_mean_diff(a, b, n_permutations=500, seed=42)
    second = permutation_test_mean_diff(a, b, n_permutations=500, seed=42)
    assert first == second, "the same seed must give the same p-value: use default_rng(seed)"


def test_rescaling_both_groups_leaves_p_unchanged():
    a, b = EXACT_CASES[0]
    p = permutation_test_mean_diff(a, b, n_permutations=2000, seed=7)
    p_scaled = permutation_test_mean_diff(
        np.asarray(a) * 10.0, np.asarray(b) * 10.0, n_permutations=2000, seed=7
    )
    assert p == p_scaled, "scaling both groups by the same factor must not change the p-value"


INVALID_CASES = [
    ("empty_first_group", [], [1.0, 2.0], 10, "an empty group has no mean: raise ValueError"),
    ("empty_second_group", [1.0, 2.0], [], 10, "an empty second group must raise ValueError"),
    ("two_dimensional", [[1.0, 2.0]], [3.0, 4.0], 10, "a 2-D group is not 1-D: raise ValueError"),
    ("nan_value", [1.0, float("nan")], [3.0, 4.0], 10, "a NaN observation must raise ValueError"),
    ("zero_permutations", [1.0, 2.0], [3.0, 4.0], 0, "n_permutations = 0 must raise ValueError"),
]


@pytest.mark.parametrize(
    ("case", "a", "b", "n_permutations", "mistake"),
    INVALID_CASES,
    ids=[case[0] for case in INVALID_CASES],
)
def test_invalid_input_raises_value_error(case, a, b, n_permutations, mistake):
    _must_raise_value_error(
        lambda: permutation_test_mean_diff(a, b, n_permutations=n_permutations, seed=0),
        mistake,
    )


def test_input_arrays_are_not_mutated():
    a = np.array([3.0, 1.0, 2.0])
    b = np.array([9.0, 7.0, 8.0, 6.0])
    a_before, b_before = a.copy(), b.copy()
    permutation_test_mean_diff(a, b, n_permutations=200, seed=5)
    np.testing.assert_array_equal(a, a_before, err_msg="group a was modified in place")
    np.testing.assert_array_equal(b, b_before, err_msg="group b was modified in place")


def test_two_dimensional_groups_are_rejected_even_when_both_are_2_d():
    _must_raise_value_error(
        lambda: permutation_test_mean_diff([[1.0, 2.0]], [[3.0, 4.0]], n_permutations=10, seed=0),
        "a 2-D group is not 1-D: raise ValueError, whatever the other group's shape",
    )


def test_a_single_permutation_is_a_valid_request():
    p = permutation_test_mean_diff([1.0, 2.0], [3.0, 4.0], n_permutations=1, seed=0)
    assert p in (0.5, 1.0), (
        "n_permutations = 1 is allowed (at least 1): with one relabelling p = (count + 1) / 2"
    )


def test_default_number_of_permutations_is_5000():
    a, b = EXACT_CASES[0]
    default = permutation_test_mean_diff(a, b, seed=11)
    explicit = permutation_test_mean_diff(a, b, n_permutations=5000, seed=11)
    assert default == explicit, (
        "the default number of relabellings is 5000: the default p-value must match "
        "n_permutations=5000 with the same seed"
    )


def test_relabelling_just_below_the_tolerance_band_is_not_counted():
    # Pooled values 0, 2e9 - 0.5, 1 and -1.5. The split {0, 1} | {2e9 - 0.5, -1.5} has a mean
    # difference 1.5 below the observed 1e9, which is more than 1e-9 relative, so it is not a tie.
    a, b = [0.0, 2e9 - 0.5], [1.0, -1.5]
    p = permutation_test_mean_diff(a, b, n_permutations=N_PERM, seed=0)
    assert abs(p - _exact_p(a, b)) <= MC_TOLERANCE, (
        "a relabelling whose mean difference is more than 1e-9 relative below the observed one "
        "is not a tie: it must not be counted as at least as extreme"
    )
