import functools

import numpy as np
import pytest
from statsmodels.stats.proportion import proportions_ztest
from submission import two_proportion_z_test

from dogma_data import load_frame, rng

# Pairs of groups of the adult census. In each pair the two arms are of unequal size and the
# higher rate is sometimes in the first arm and sometimes in the second.
GROUP_PAIRS = [
    ("sex", "Male", "Female"),
    ("race", "Black", "White"),
    ("marital.status", "Married-civ-spouse", "Never-married"),
    ("education", "HS-grad", "Bachelors"),
    ("workclass", "Private", "Self-emp-not-inc"),
    ("relationship", "Wife", "Husband"),
    ("occupation", "Exec-managerial", "Prof-specialty"),
    ("education", "Masters", "Doctorate"),
]


@functools.cache
def _oracle_cases():
    """Real counts (x_a, n_a, x_b, n_b): people earning above 50K in two census groups.

    Nine tenths of the 6,000 people of the adult table are drawn, and each pair of groups
    gives the number of people and the number of high earners in each arm.
    """
    adult = load_frame("adult")
    adult = adult.iloc[rng("two-proportion-z-test-p02:census").permutation(len(adult))[:5400]]
    high = adult["income"] == ">50K"
    cases = []
    for column, first, second in GROUP_PAIRS:
        in_a, in_b = adult[column] == first, adult[column] == second
        cases.append(
            (int(high[in_a].sum()), int(in_a.sum()), int(high[in_b].sum()), int(in_b.sum()))
        )
    return cases


def _reference(x_a, n_a, x_b, n_b):
    """Independent reference: statsmodels with arm B listed first, so z is (p_B - p_A) / SE."""
    z_stat, p_value = proportions_ztest(count=np.array([x_b, x_a]), nobs=np.array([n_b, n_a]))
    return [float(z_stat), float(p_value)]


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
    "case", range(len(GROUP_PAIRS)), ids=[f"oracle{index}" for index in range(len(GROUP_PAIRS))]
)
def test_matches_statsmodels_pooled_z_test(case):
    case = _oracle_cases()[case]
    np.testing.assert_allclose(
        two_proportion_z_test(*case),
        _reference(*case),
        rtol=1e-9,
        atol=1e-12,
        err_msg="z or p differs from the pooled two-proportion test: check pooled rate and SE",
    )


def test_swapping_the_arms_flips_only_the_sign_of_z():
    x_a, n_a, x_b, n_b = 37, 250, 58, 310
    z_ab, p_ab = two_proportion_z_test(x_a, n_a, x_b, n_b)
    z_ba, p_ba = two_proportion_z_test(x_b, n_b, x_a, n_a)
    np.testing.assert_allclose(
        [z_ba, p_ba],
        [-z_ab, p_ab],
        rtol=1e-9,
        err_msg="swapping the arms must negate z and keep p: check the sign convention",
    )


def test_doubling_both_arms_multiplies_z_by_root_two():
    x_a, n_a, x_b, n_b = 37, 250, 58, 310
    z_one, p_one = two_proportion_z_test(x_a, n_a, x_b, n_b)
    z_two, p_two = two_proportion_z_test(2 * x_a, 2 * n_a, 2 * x_b, 2 * n_b)
    np.testing.assert_allclose(
        z_two,
        np.sqrt(2.0) * z_one,
        rtol=1e-9,
        err_msg="doubling both arms keeps the rates: z must grow by sqrt(2) with the SE",
    )
    assert p_two < p_one, "more users at the same rates must give a smaller p-value"


def test_tiny_two_sided_p_value_is_not_rounded_to_zero():
    case = (50000, 100000, 56000, 100000)
    z_stat, p_value = two_proportion_z_test(*case)
    ref_z, ref_p = _reference(*case)
    assert p_value > 0.0, (
        "p-value underflowed to zero: use the normal survival function, not 1 minus the CDF"
    )
    np.testing.assert_allclose(
        [z_stat, p_value],
        [ref_z, ref_p],
        rtol=1e-6,
        err_msg="tiny p-value differs from the reference: compute the tail with the survival fn",
    )


def test_equal_rates_in_unequal_arms_give_z_zero_and_p_one():
    z_stat, p_value = two_proportion_z_test(30, 100, 60, 200)
    np.testing.assert_allclose(
        [z_stat, p_value],
        [0.0, 1.0],
        atol=1e-12,
        err_msg="identical rates should give z = 0 and p = 1 even when the arms differ in size",
    )


def test_numpy_integer_counts_give_the_same_result():
    plain = two_proportion_z_test(120, 1000, 90, 1000)
    numpy_counts = two_proportion_z_test(
        np.int64(120), np.int64(1000), np.int64(90), np.int64(1000)
    )
    np.testing.assert_allclose(
        numpy_counts,
        plain,
        rtol=0,
        atol=0,
        err_msg="NumPy integer counts must be accepted and give the same result",
    )


def test_returns_two_float_values():
    result = two_proportion_z_test(12, 80, 30, 95)
    assert len(result) == 2, "return exactly (z, p)"
    assert all(isinstance(value, float) for value in result), "z and p should be floats"


INVALID_CASES = [
    ("empty_arm_a", (0, 0, 5, 10), "an arm with no users has no rate: raise ValueError"),
    ("empty_arm_b", (5, 10, 0, 0), "an empty arm B must raise ValueError"),
    (
        "conversions_above_users",
        (11, 10, 5, 10),
        "more conversions than users must raise ValueError",
    ),
    ("negative_conversions", (-1, 10, 5, 10), "negative conversions must raise ValueError"),
    ("pooled_rate_zero", (0, 10, 0, 10), "a pooled rate of 0 leaves SE at zero: raise ValueError"),
    ("pooled_rate_one", (10, 10, 5, 5), "a pooled rate of 1 leaves SE at zero: raise ValueError"),
    ("float_count", (5.5, 10, 4, 10), "a non-integer count must raise ValueError"),
    ("boolean_count", (True, 10, 4, 10), "a boolean is not a count: raise ValueError"),
]


@pytest.mark.parametrize(
    ("case", "args", "mistake"),
    INVALID_CASES,
    ids=[case[0] for case in INVALID_CASES],
)
def test_invalid_input_raises_value_error(case, args, mistake):
    _must_raise_value_error(lambda: two_proportion_z_test(*args), mistake)


def _must_accept(args, mistake):
    """Fail with the kind of mistake when a valid count is refused."""
    try:
        result = two_proportion_z_test(*args)
    except Exception as exc:
        pytest.fail(f"{mistake} (raised {type(exc).__name__} instead)")
    if not (isinstance(result, tuple) and len(result) == 2):
        pytest.fail("return the pair (z, p) for valid counts")


def test_zero_conversions_in_arm_a_is_accepted():
    _must_accept(
        (0, 100, 10, 100),
        "zero conversions in arm A is a valid count (0 <= x_a): do not reject it",
    )


def test_zero_conversions_in_arm_b_is_accepted():
    _must_accept(
        (10, 100, 0, 100),
        "zero conversions in arm B is a valid count (0 <= x_b): do not reject it",
    )


def test_every_user_converting_in_arm_a_is_accepted():
    _must_accept(
        (100, 100, 10, 100),
        "all users converting in arm A is a valid count (x_a <= n_a): do not reject it",
    )


def test_every_user_converting_in_arm_b_is_accepted():
    _must_accept(
        (30, 100, 50, 50),
        "all users converting in arm B is a valid count (x_b <= n_b): do not reject it",
    )


def test_single_user_in_arm_a_is_accepted():
    _must_accept(
        (1, 1, 0, 10),
        "an arm with one user is valid (n_a >= 1): do not require more than one user",
    )


def test_single_user_in_arm_b_is_accepted():
    _must_accept(
        (10, 100, 1, 1),
        "an arm with one user is valid (n_b >= 1): do not require more than one user",
    )


def test_negative_conversions_in_arm_b_raise_value_error():
    _must_raise_value_error(
        lambda: two_proportion_z_test(10, 100, -1, 100),
        "a negative conversion count in arm B was accepted: counts must lie in [0, n]",
    )


def test_whole_valued_float_count_is_rejected():
    # Floats are not counts, even when the value is whole: 120.0 is a float, so it must raise.
    _must_raise_value_error(
        lambda: two_proportion_z_test(120.0, 1000, 90, 1000),
        "a whole-valued float count was accepted: floats must raise ValueError",
    )
