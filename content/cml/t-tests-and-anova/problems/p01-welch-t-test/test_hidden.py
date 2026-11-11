import numpy as np
import pytest
from scipy import stats
from submission import welch_t_test

from dogma_data import load

# (seed, n_a, n_b, spread_a, spread_b, shift_b): unequal sizes and unequal spreads throughout.
ORACLE_CASES = [
    (0, 5, 9, 1.0, 3.0, 0.5),
    (1, 8, 4, 2.0, 0.5, 1.0),
    (2, 12, 30, 1.0, 1.0, 0.0),
    (3, 3, 20, 0.2, 2.5, -1.0),
    (4, 30, 6, 4.0, 1.0, 2.0),
    (5, 2, 15, 1.0, 0.1, 0.3),
    (6, 25, 25, 1.5, 2.5, 0.8),
    (7, 6, 11, 0.5, 6.0, -0.4),
]


def _seeded_pair(seed, n_a, n_b, *, spread_a, spread_b, shift_b):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, spread_a, size=n_a)
    b = rng.normal(shift_b, spread_b, size=n_b)
    return a, b


def _reference(a, b):
    """Independent reference: scipy's Welch test, with df read from the result object."""
    result = stats.ttest_ind(np.asarray(a, float), np.asarray(b, float), equal_var=False)
    return [float(result.statistic), float(result.df), float(result.pvalue)]


def _must_raise_value_error(call, mistake):
    """Fail with the kind of mistake, instead of pytest's generic 'DID NOT RAISE'."""
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    pytest.fail(mistake)


@pytest.mark.parametrize("case", ORACLE_CASES, ids=list("abcdefgh"))
def test_matches_scipy_welch_on_seeded_samples(case):
    seed, n_a, n_b, spread_a, spread_b, shift_b = case
    a, b = _seeded_pair(seed, n_a, n_b, spread_a=spread_a, spread_b=spread_b, shift_b=shift_b)
    np.testing.assert_allclose(
        welch_t_test(a, b),
        _reference(a, b),
        rtol=1e-9,
        atol=1e-12,
        err_msg="t, df or p differs from scipy's Welch test: check the variance terms and the df",
    )


def test_equal_sizes_and_spreads_give_welch_df_of_two_n_minus_one():
    rng = np.random.default_rng(20)
    a = rng.normal(0.0, 1.0, size=9)
    b = rng.normal(0.0, 1.0, size=9)
    b = (b - b.mean()) / b.std(ddof=1) * a.std(ddof=1) + 0.7  # same spread, shifted mean
    t_stat, df, _ = welch_t_test(a, b)
    pooled_t = (a.mean() - b.mean()) / np.sqrt(
        ((8 * a.var(ddof=1) + 8 * b.var(ddof=1)) / 16) * (2 / 9)
    )
    np.testing.assert_allclose(
        df,
        16.0,
        rtol=1e-9,
        err_msg="with equal sizes and spreads the Welch df must equal n_a + n_b - 2 = 16",
    )
    np.testing.assert_allclose(
        t_stat,
        pooled_t,
        rtol=1e-9,
        err_msg="with equal sizes and spreads the Welch t must equal the pooled t",
    )


def test_unequal_spreads_need_welch_df_not_pooled_df():
    # Student's pooled test would call this significant (p near 0.049). Welch's test does not.
    a = [0.4, 0.2, -0.3, -0.1, -0.3, -0.1, -0.2, -0.5, 0.4, 0.6, 0.0, 0.5, 0.3, 0.5, 0.1]
    b = [2.7, -3.2, 11.3]
    t_stat, df, p_value = welch_t_test(a, b)
    assert df < 5.0, (
        "unequal spreads need Welch's degrees of freedom (about 2 here), not n_a + n_b - 2 = 16"
    )
    np.testing.assert_allclose(
        [t_stat, df, p_value],
        _reference(a, b),
        rtol=1e-9,
        err_msg="the Welch result differs from scipy on the unequal-spread sample",
    )


def test_tiny_two_sided_p_value_is_not_rounded_to_zero():
    # A large separation gives |t| near 30, so the two-sided p-value is far below double precision
    # relative to 1. 1 minus the CDF rounds it to exactly zero.
    rng = np.random.default_rng(50)
    a = rng.normal(0.0, 1.0, size=30)
    b = rng.normal(10.0, 1.0, size=30)
    _, _, p_value = welch_t_test(a, b)
    _, _, ref_p = _reference(a, b)
    assert p_value > 0.0, (
        "p-value underflowed to zero: use the survival function of Student's t, not 1 minus the CDF"
    )
    np.testing.assert_allclose(
        p_value,
        ref_p,
        rtol=1e-6,
        err_msg="tiny p-value differs from the reference: compute the two-sided tail directly",
    )


def test_one_constant_sample_is_valid_and_matches_scipy():
    a = [3.0, 3.0, 3.0, 3.0]
    b = [1.0, 2.5, 4.0, 2.0, 6.0]
    np.testing.assert_allclose(
        welch_t_test(a, b),
        _reference(a, b),
        rtol=1e-9,
        err_msg="one constant sample has a positive standard error: the result must match scipy",
    )


def test_swapping_groups_flips_t_and_keeps_df_and_p():
    a, b = _seeded_pair(30, 7, 12, spread_a=1.0, spread_b=2.0, shift_b=0.4)
    t_ab, df_ab, p_ab = welch_t_test(a, b)
    t_ba, df_ba, p_ba = welch_t_test(b, a)
    np.testing.assert_allclose(
        [t_ba, df_ba, p_ba],
        [-t_ab, df_ab, p_ab],
        rtol=1e-9,
        err_msg="swapping the groups must flip the sign of t and keep df and the two-sided p",
    )


def test_shifting_and_rescaling_both_samples_leaves_p_and_df_unchanged():
    a, b = _seeded_pair(31, 6, 10, spread_a=2.0, spread_b=1.0, shift_b=1.0)
    base = welch_t_test(a, b)
    shifted = welch_t_test(a + 5.0, b + 5.0)
    scaled = welch_t_test(a * 4.0, b * 4.0)
    np.testing.assert_allclose(
        shifted,
        base,
        rtol=1e-9,
        err_msg="adding the same constant to both samples must not change the test",
    )
    np.testing.assert_allclose(
        scaled,
        [base[0], base[1], base[2]],
        rtol=1e-9,
        err_msg="multiplying both samples by one factor must not change t, df or p",
    )


@pytest.mark.parametrize("magnitude", [1e-150, 1e150], ids=["tiny-magnitude", "huge-magnitude"])
def test_extreme_magnitudes_do_not_underflow_or_overflow(magnitude):
    a, b = _seeded_pair(32, 5, 7, spread_a=1.0, spread_b=2.0, shift_b=0.5)
    expected = _reference(a, b)
    np.testing.assert_allclose(
        welch_t_test(a * magnitude, b * magnitude),
        expected,
        rtol=1e-8,
        err_msg="data at tiny or huge magnitude gave a different result: rescale before squaring",
    )


def test_large_common_offset_keeps_the_spread():
    rng = np.random.default_rng(33)
    a = 1e8 + rng.normal(0.0, 1.0, size=8)
    b = 1e8 + rng.normal(0.5, 2.0, size=10)
    np.testing.assert_allclose(
        welch_t_test(a, b),
        _reference(a, b),
        rtol=1e-6,
        err_msg="a large common offset destroys the spread: centre the data before the variances",
    )


def test_integer_input_returns_three_float_values():
    result = welch_t_test([1, 2, 3, 4], [2, 3, 4, 5, 6])
    assert len(result) == 3, "return exactly (t, df, p)"
    assert all(isinstance(value, float) for value in result), (
        "all three values should be floats, even for integer input"
    )
    np.testing.assert_allclose(
        result,
        _reference([1, 2, 3, 4], [2, 3, 4, 5, 6]),
        rtol=1e-9,
        err_msg="integer input must give the same three values as float input",
    )


INVALID_CASES = [
    (
        "two_dimensional",
        [[1.0, 2.0], [3.0, 4.0]],
        [1.0, 2.0],
        "a 2-D sample is not 1-D: raise ValueError",
    ),
    ("empty_group", [], [1.0, 2.0, 3.0], "an empty sample has no mean: raise ValueError"),
    (
        "single_observation",
        [1.0],
        [2.0, 3.0, 4.0],
        "one observation has no variance: raise ValueError",
    ),
    (
        "nan_value",
        [1.0, float("nan"), 3.0],
        [1.0, 2.0, 4.0],
        "a NaN observation must raise ValueError",
    ),
    (
        "infinite_value",
        [1.0, float("inf"), 3.0],
        [1.0, 2.0, 4.0],
        "an infinite observation must raise ValueError",
    ),
    (
        "both_constant",
        [2.0, 2.0, 2.0],
        [5.0, 5.0],
        "both samples constant: t is undefined, raise ValueError",
    ),
]


@pytest.mark.parametrize(
    ("case", "a", "b", "mistake"),
    INVALID_CASES,
    ids=[case[0] for case in INVALID_CASES],
)
def test_invalid_input_raises_value_error(case, a, b, mistake):
    _must_raise_value_error(lambda: welch_t_test(a, b), mistake)


def test_input_arrays_are_not_mutated():
    a = np.array([3.0, 1.0, 2.0, 5.0])
    b = np.array([9.0, 7.0, 8.0])
    a_before, b_before = a.copy(), b.copy()
    welch_t_test(a, b)
    np.testing.assert_array_equal(a, a_before, err_msg="sample a was modified in place")
    np.testing.assert_array_equal(b, b_before, err_msg="sample b was modified in place")


def test_two_dimensional_samples_raise_even_when_shapes_match():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[1.0, 2.0], [3.0, 5.0]]
    _must_raise_value_error(
        lambda: welch_t_test(a, b),
        "a 2-D sample must raise ValueError: each sample must be 1-D, even when both "
        "have the same shape",
    )


def test_two_constant_samples_raise_when_their_means_are_not_exact():
    cases = [
        ([0.1, 0.1, 0.1], [0.7, 0.7, 0.7, 0.7]),
        ([0.3] * 7, [0.1] * 5),
        ([1.1] * 3, [2.2] * 4),
        ([0.1, 0.1], [0.2] * 5),
        ([0.7, 0.7], [0.1] * 5),
    ]
    for a, b in cases:
        _must_raise_value_error(
            lambda a=a, b=b: welch_t_test(a, b),
            "two constant samples must raise ValueError even when their float mean is inexact: "
            "the t statistic is undefined",
        )


@pytest.fixture(scope="module")
def iris_sepal_lengths():
    # Sepal lengths in cm (one decimal) for the setosa and virginica species, 50 rows each.
    table = load("iris")
    return table.X[:, 0], table.y


def test_matches_scipy_on_float32_storage_of_iris_setosa_and_virginica(iris_sepal_lengths):
    values, species = iris_sepal_lengths
    a = values[species == 0].astype(np.float32)  # real measurements stored as float32
    b = values[species == 2].astype(np.float32)
    np.testing.assert_allclose(
        welch_t_test(a, b),
        _reference(a.astype(np.float64), b.astype(np.float64)),
        rtol=1e-9,
        atol=1e-12,
        err_msg="float32 input gives float32-rounded statistics: convert to float64 first",
    )


def test_result_is_a_tuple_of_three_values():
    result = welch_t_test([1.0, 2.5, 4.0, 3.0], [7.0, 9.5, 8.0, 6.5, 10.0])
    assert isinstance(result, tuple), (
        "the result must be a tuple (t, df, p), not an array or a list"
    )
    assert len(result) == 3, "the result must hold exactly three values: t, df and p"


def test_constant_first_sample_at_overall_mean_still_gives_a_result():
    # Both group means are 1, so the mean difference is zero. Only sample b varies.
    # Hand check: vb = var([0, 1, 2], ddof=1) / 3 = 1/3, va = 0, so df = (1/3)^2 / ((1/3)^2 / 2).
    try:
        t, df, p = welch_t_test([1.0, 1.0], [0.0, 1.0, 2.0])
    except ValueError:
        pytest.fail(
            "a constant sample whose value equals the overall mean is valid when the other "
            "sample varies: it must give a result, not an error"
        )
    assert t == pytest.approx(0.0, abs=1e-12), (
        "a constant first sample at the overall mean gave the wrong t statistic"
    )
    assert df == pytest.approx(2.0, rel=1e-9), (
        "a constant first sample gave the wrong degrees of freedom: only the varying "
        "sample's variance term should count"
    )
    assert p == pytest.approx(1.0, rel=1e-9), (
        "a constant first sample at the overall mean gave the wrong two-sided p-value"
    )


def test_constant_second_sample_at_overall_mean_still_gives_a_result():
    # Both group means are 1, so the mean difference is zero. Only sample a varies.
    # Hand check: va = var([0, 2, 1], ddof=1) / 3 = 1/3, vb = 0, so df = (1/3)^2 / ((1/3)^2 / 2).
    try:
        t, df, p = welch_t_test([0.0, 2.0, 1.0], [1.0, 1.0, 1.0])
    except ValueError:
        pytest.fail(
            "a constant sample whose value equals the overall mean is valid when the other "
            "sample varies: it must give a result, not an error"
        )
    assert t == pytest.approx(0.0, abs=1e-12), (
        "a constant second sample at the overall mean gave the wrong t statistic"
    )
    assert df == pytest.approx(2.0, rel=1e-9), (
        "a constant second sample gave the wrong degrees of freedom: only the varying "
        "sample's variance term should count"
    )
    assert p == pytest.approx(1.0, rel=1e-9), (
        "a constant second sample at the overall mean gave the wrong two-sided p-value"
    )


def test_constant_sample_at_overall_mean_with_two_point_partner_still_gives_a_result():
    # Both group means are 1, so the mean difference is zero. Sample b is two points, 0 and 2.
    # Hand check: vb = var([0, 2], ddof=1) / 2 = 1, va = 0, so df = 1 / (1 / 1).
    try:
        t, df, p = welch_t_test([1.0, 1.0], [0.0, 2.0])
    except ValueError:
        pytest.fail(
            "a constant sample whose value equals the overall mean is valid when the other "
            "sample varies: it must give a result, not an error"
        )
    assert t == pytest.approx(0.0, abs=1e-12), (
        "a constant sample at the overall mean with a varying partner gave the wrong t statistic"
    )
    assert df == pytest.approx(1.0, rel=1e-9), (
        "a constant sample at the overall mean gave the wrong degrees of freedom: only the "
        "varying sample's variance term should count"
    )
    assert p == pytest.approx(1.0, rel=1e-9), (
        "a constant sample at the overall mean gave the wrong two-sided p-value"
    )
