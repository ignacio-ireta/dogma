import numpy as np
import pytest
from scipy import stats
from submission import one_way_anova

from dogma_data import load, sample_rows
from dogma_data import rng as data_rng

# Three groups whose F is about 225: the upper-tail p-value is about 1e-27, far below where 1 - cdf
# still carries any information.
TAIL_GROUPS = [
    [
        0.42,
        -1.49,
        -0.15,
        0.72,
        -0.62,
        0.03,
        0.75,
        -0.58,
        0.41,
        0.95,
        0.22,
        0.81,
        -0.07,
        -0.21,
        0.23,
        -0.72,
        0.55,
        0.02,
        -0.65,
        0.57,
    ],
    [
        2.34,
        1.84,
        1.69,
        2.19,
        1.21,
        1.42,
        2.64,
        2.18,
        2.63,
        2.12,
        2.87,
        1.5,
        2.64,
        2.52,
        2.64,
        2.86,
        1.88,
        2.08,
        3.08,
        2.72,
    ],
    [
        4.58,
        3.37,
        5.04,
        4.33,
        4.54,
        3.2,
        4.23,
        2.9,
        4.47,
        4.08,
        4.19,
        3.57,
        4.22,
        3.99,
        3.2,
        4.2,
        3.95,
        3.31,
        4.56,
        3.36,
    ],
]

# (seed, group sizes): several group counts and unequal sizes.
ORACLE_CASES = [
    (0, (5, 9)),
    (1, (4, 6, 8)),
    (2, (3, 10, 7, 12)),
    (3, (20, 3, 11)),
    (4, (2, 2, 2)),
    (5, (30, 25)),
    (6, (6, 6, 6, 6, 6)),
    (7, (1, 5, 9)),
]


def _seeded_groups(seed, sizes):
    """Real groups: the mean pixel intensity of digits images, one group per digit.

    Group i holds sizes[i] images of the digit i. The digits differ in how much ink they use
    and in how much that varies, so the groups have unequal means and spreads.
    """
    X, digit = sample_rows("digits", data_rng(f"anova-p02:groups:{seed}"), 1797)
    ink = X.mean(axis=1)
    return [ink[digit == float(i)][:size] for i, size in enumerate(sizes)]


def _reference(groups):
    """Independent reference: scipy's one-way ANOVA."""
    result = stats.f_oneway(*[np.asarray(g, dtype=np.float64) for g in groups])
    return [float(result.statistic), float(result.pvalue)]


def _must_raise_value_error(call, mistake):
    """Fail with the kind of mistake, instead of pytest's generic 'DID NOT RAISE'."""
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


@pytest.mark.parametrize("case", ORACLE_CASES, ids=[f"seed{c[0]}" for c in ORACLE_CASES])
def test_matches_scipy_one_way_anova_on_seeded_groups(case):
    seed, sizes = case
    groups = _seeded_groups(seed, sizes)
    np.testing.assert_allclose(
        one_way_anova(*groups),
        _reference(groups),
        rtol=1e-9,
        atol=1e-12,
        err_msg="F or p differs from the one-way ANOVA reference: check SSB, SSW and both df",
    )


def test_two_groups_give_f_equal_to_the_square_of_the_pooled_t():
    a, b = _seeded_groups(40, (7, 11))
    na, nb = a.size, b.size
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    t_pooled = (a.mean() - b.mean()) / np.sqrt(sp2 * (1 / na + 1 / nb))
    f_stat, _ = one_way_anova(a, b)
    np.testing.assert_allclose(
        f_stat,
        t_pooled**2,
        rtol=1e-9,
        err_msg="with two groups F must equal the square of the pooled t: check df1 = k - 1",
    )


def test_tiny_upper_tail_p_value_is_not_rounded_to_zero():
    f_stat, p_value = one_way_anova(*TAIL_GROUPS)
    ref_f, ref_p = _reference(TAIL_GROUPS)
    assert p_value > 0.0, (
        "p-value underflowed to zero: use the survival function of F, not 1 minus the CDF"
    )
    np.testing.assert_allclose(
        [f_stat, p_value],
        [ref_f, ref_p],
        rtol=1e-6,
        err_msg="tiny p-value differs from the reference: compute the upper tail directly",
    )


def test_singleton_group_is_allowed_and_matches_scipy():
    groups = [[5.0], [1.0, 2.0, 3.0, 4.0]]
    np.testing.assert_allclose(
        one_way_anova(*groups),
        _reference(groups),
        rtol=1e-9,
        err_msg="a group of one value is valid when N > k: check the within df and the SSW",
    )


def test_shifting_rescaling_and_reordering_leave_the_result_unchanged():
    groups = _seeded_groups(41, (6, 9, 5))
    base = one_way_anova(*groups)
    np.testing.assert_allclose(
        one_way_anova(*[g + 7.5 for g in groups]),
        base,
        rtol=1e-9,
        err_msg="adding one constant to every value must not change F or p",
    )
    np.testing.assert_allclose(
        one_way_anova(*[g * 3.0 for g in groups]),
        base,
        rtol=1e-9,
        err_msg="multiplying every value by one factor must not change F or p",
    )
    np.testing.assert_allclose(
        one_way_anova(groups[2], groups[0], groups[1]),
        base,
        rtol=1e-9,
        err_msg="the order of the groups must not change F or p",
    )


def test_large_common_offset_keeps_the_within_group_spread():
    groups = [g + 1e8 for g in _seeded_groups(42, (8, 8, 8))]
    np.testing.assert_allclose(
        one_way_anova(*groups),
        _reference(groups),
        rtol=1e-6,
        err_msg="a large common offset destroys the within-group spread: use deviations from means",
    )


def test_returns_two_float_values():
    result = one_way_anova([1, 2, 3, 4], [2, 3, 4, 5, 6])
    assert len(result) == 2, "return exactly (F, p)"
    assert all(isinstance(value, float) for value in result), (
        "F and p should be floats, even for integer input"
    )


INVALID_CASES = [
    ("single_group", ([1.0, 2.0, 3.0],), "fewer than two groups must raise ValueError"),
    ("empty_group", ([1.0, 2.0], []), "an empty group has no mean: raise ValueError"),
    (
        "two_dimensional",
        ([[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0]),
        "a 2-D group is not 1-D: raise ValueError",
    ),
    (
        "nan_value",
        ([1.0, float("nan"), 3.0], [1.0, 2.0, 4.0]),
        "a NaN observation must raise ValueError",
    ),
    ("no_within_df", ([1.0], [2.0]), "N = k leaves no within-group df: raise ValueError"),
    (
        "zero_within_variation",
        ([1.0, 1.0], [2.0, 2.0]),
        "zero within-group variation: F is undefined, raise ValueError",
    ),
]


@pytest.mark.parametrize(
    ("case", "groups", "mistake"),
    INVALID_CASES,
    ids=[case[0] for case in INVALID_CASES],
)
def test_invalid_input_raises_value_error(case, groups, mistake):
    _must_raise_value_error(lambda: one_way_anova(*groups), mistake)


def test_input_arrays_are_not_mutated():
    a = np.array([3.0, 1.0, 2.0])
    b = np.array([9.0, 7.0, 8.0, 6.0])
    a_before, b_before = a.copy(), b.copy()
    one_way_anova(a, b)
    np.testing.assert_array_equal(a, a_before, err_msg="group a was modified in place")
    np.testing.assert_array_equal(b, b_before, err_msg="group b was modified in place")


def test_empty_group_raises_even_with_enough_other_observations():
    _must_raise_value_error(
        lambda: one_way_anova([1.0, 2.0, 3.0], []),
        "an empty group must raise ValueError, even when the other groups leave within-group df",
    )


def test_two_dimensional_groups_raise_even_when_shapes_match():
    _must_raise_value_error(
        lambda: one_way_anova([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 5.0]]),
        "a 2-D group must raise ValueError: each group must be 1-D, even when the shapes match",
    )


def test_one_more_observation_than_groups_is_valid_and_matches_scipy():
    groups = [[1.0, 2.0], [4.0]]
    np.testing.assert_allclose(
        one_way_anova(*groups),
        _reference(groups),
        rtol=1e-9,
        err_msg="N = k + 1 leaves one within-group df, which is valid: only N <= k must raise",
    )


@pytest.fixture(scope="module")
def iris_species_sepal_lengths():
    # Sepal lengths of the three iris species (50 rows each), in cm to one decimal.
    table = load("iris")
    return [table.X[table.y == k, 0] for k in (0, 1, 2)]


def test_matches_scipy_on_float32_storage_of_iris_species(iris_species_sepal_lengths):
    stored = [group.astype(np.float32) for group in iris_species_sepal_lengths]
    np.testing.assert_allclose(
        one_way_anova(*stored),
        _reference([group.astype(np.float64) for group in stored]),
        rtol=1e-9,
        atol=1e-12,
        err_msg="float32 groups give float32-rounded F and p: convert each group to float64 first",
    )


def test_result_is_a_tuple_of_two_values():
    result = one_way_anova([2.0, 3.5, 2.5], [6.0, 7.5, 5.5, 6.5])
    assert isinstance(result, tuple), "the result must be a tuple (F, p), not an array or a list"
    assert len(result) == 2, "the result must hold exactly two values: F and p"
