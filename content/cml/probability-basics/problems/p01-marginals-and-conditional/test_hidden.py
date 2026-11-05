import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import marginals_and_conditional

SEEDS = st.integers(min_value=0, max_value=10**6)
SHAPES = st.tuples(st.integers(min_value=1, max_value=4), st.integers(min_value=1, max_value=4))
ORACLE_HINT = (
    "marginals or conditional differ from the pandas reference: P(X) sums over the columns, "
    "P(Y) over the rows, row i of P(Y | X) is P(X=i, Y=j) / P(X=i), not a copy of P(Y), and a "
    "zero-probability row is NaN"
)


def _random_table(seed, m, k):
    """Random joint table. About a quarter of the rows are zeroed, so some rows have zero mass."""
    rng = np.random.default_rng(seed)
    raw = rng.random((m, k)) ** 2
    raw[rng.random(m) < 0.25] = 0.0
    if raw.sum() == 0.0:
        raw[0, 0] = 1.0
    return raw / raw.sum()


def _fail(why):
    """The one place these tests raise, so a Hypothesis run reports a single named failure."""
    raise AssertionError(why)


def _outputs(joint, what):
    """Return (outputs, None), or (None, the mistake). Never raises, so each test has one site."""
    try:
        return marginals_and_conditional(joint), None
    except ValueError:
        return None, (
            "a valid table (entries >= 0, sum 1 up to float rounding) was rejected: the sum or "
            "sign check is too strict; a sum within 1e-9 of 1 must be accepted"
        )
    except Exception as exc:
        return None, f"{what} (the function raised {type(exc).__name__})"


def _solve(joint, what):
    """Call the function on a table that must be accepted. A crash is reported as a mistake."""
    outputs, problem = _outputs(joint, what)
    if problem is not None:
        _fail(problem)
    return outputs


def _expect_value_error(joint, why):
    try:
        marginals_and_conditional(joint)
    except ValueError:
        return
    except Exception as exc:
        _fail(f"{why} (the function raised {type(exc).__name__} instead of ValueError)")
    _fail(why)


def _pandas_reference(joint):
    """Independent route: a long DataFrame, groupby for the marginals, crosstab for conditionals."""
    m, k = joint.shape
    long = pd.DataFrame(
        [(i, j, joint[i, j]) for i in range(m) for j in range(k)], columns=["x", "y", "p"]
    )
    p_x = long.groupby("x")["p"].sum().reindex(range(m)).to_numpy()
    p_y = long.groupby("y")["p"].sum().reindex(range(k)).to_numpy()
    cond = (
        pd.crosstab(long["x"], long["y"], values=long["p"], aggfunc="sum", normalize="index")
        .reindex(index=range(m), columns=range(k))
        .to_numpy(copy=True)  # pandas 3 may hand back a read-only view; we write into it below
    )
    # crosstab reports an all-zero row as 0.0; the contract says such a row is undefined (NaN).
    cond[p_x == 0] = np.nan
    return p_x, p_y, cond


@pytest.mark.parametrize(
    ("m", "k", "seed"), [(2, 3, 1), (3, 2, 2), (1, 4, 3), (4, 1, 4), (3, 3, 5), (4, 4, 6)]
)
def test_matches_pandas_marginals_and_crosstab_on_random_tables(m, k, seed):
    joint = _random_table(seed, m, k)
    p_x, p_y, cond = _solve(joint, ORACLE_HINT)
    ref_x, ref_y, ref_cond = _pandas_reference(joint)
    np.testing.assert_allclose(
        p_x, ref_x, rtol=1e-9, atol=1e-12, err_msg="marginal P(X) differs: " + ORACLE_HINT
    )
    np.testing.assert_allclose(
        p_y, ref_y, rtol=1e-9, atol=1e-12, err_msg="marginal P(Y) differs: " + ORACLE_HINT
    )
    np.testing.assert_allclose(
        cond,
        ref_cond,
        rtol=1e-9,
        atol=1e-12,
        equal_nan=True,
        err_msg="conditional P(Y | X) differs: " + ORACLE_HINT,
    )


@settings(max_examples=80, deadline=None)
@given(SEEDS, SHAPES)
def test_each_defined_conditional_row_sums_to_one(seed, shape):
    joint = _random_table(seed, *shape)
    outputs, problem = _outputs(
        joint, "each defined row of P(Y | X) must sum to 1: the table could not be processed"
    )
    if problem is None:
        _, _, cond = outputs
        defined = joint.sum(axis=1) > 0
        if not np.allclose(cond[defined].sum(axis=1), 1.0, rtol=1e-9, atol=0.0):
            problem = (
                "each defined conditional row P(Y | X=i) must sum to 1: the division used the "
                "wrong denominator (a column sum, not the row sum)"
            )
    assert problem is None, problem


@settings(max_examples=80, deadline=None)
@given(SEEDS, SHAPES)
def test_each_marginal_sums_to_one(seed, shape):
    joint = _random_table(seed, *shape)
    outputs, problem = _outputs(joint, "marginals must be sums of the joint over one axis each")
    if problem is None:
        p_x, p_y, _ = outputs
        if not np.allclose(p_x.sum(), 1.0, rtol=1e-9, atol=0.0):
            problem = "p_x must be a distribution: its entries sum to 1"
        elif not np.allclose(p_y.sum(), 1.0, rtol=1e-9, atol=0.0):
            problem = "p_y must be a distribution: its entries sum to 1"
    assert problem is None, problem


@settings(max_examples=80, deadline=None)
@given(SEEDS, SHAPES)
def test_marginal_times_conditional_rebuilds_the_joint(seed, shape):
    joint = _random_table(seed, *shape)
    outputs, problem = _outputs(
        joint, "P(X=i) * P(Y | X=i) must rebuild the joint, so the outputs fit"
    )
    if problem is None:
        p_x, _, cond = outputs
        rebuilt = p_x[:, None] * np.nan_to_num(cond, nan=0.0)
        if not np.allclose(rebuilt, joint, rtol=1e-9, atol=1e-12):
            problem = (
                "P(X=i) * P(Y=j | X=i) must give back the joint cell P(X=i, Y=j): the conditional "
                "is scaled by the wrong factor"
            )
    assert problem is None, problem


@pytest.mark.parametrize(("m", "k", "seed"), [(2, 3, 11), (3, 2, 12), (4, 4, 13)])
def test_independent_table_has_conditional_equal_to_marginal(m, k, seed):
    rng = np.random.default_rng(seed)
    px = rng.random(m)
    px /= px.sum()
    py = rng.random(k)
    py /= py.sum()
    p_x, p_y, cond = _solve(np.outer(px, py), "independent table: outputs must be computed")
    np.testing.assert_allclose(p_x, px, rtol=1e-9, err_msg="p_x wrong for an outer-product table")
    np.testing.assert_allclose(p_y, py, rtol=1e-9, err_msg="p_y wrong for an outer-product table")
    np.testing.assert_allclose(
        cond,
        np.tile(py, (m, 1)),
        rtol=1e-9,
        atol=1e-12,
        err_msg="independent X and Y: P(Y | X=i) must equal P(Y) in every row; the conditional "
        "is not computed from the joint",
    )


def test_zero_probability_row_is_nan_while_other_rows_stay_valid():
    _, _, cond = _solve(
        [[0.0, 0.0, 0.0], [0.1, 0.2, 0.7]], "a zero-probability row must not break other rows"
    )
    assert np.isnan(cond[0]).all(), (
        "a zero-probability row must be NaN, not 0 or a made-up distribution"
    )
    np.testing.assert_allclose(
        cond[1],
        [0.1, 0.2, 0.7],
        atol=1e-12,
        err_msg="a valid row must be unaffected by a zero-probability row",
    )


def test_tiny_but_positive_marginal_still_defines_its_row():
    _, _, cond = _solve(
        np.array([[1e-200, 0.0], [0.0, 1.0]]),
        "a tiny marginal must still define its row: the zero test must be p_x > 0",
    )
    np.testing.assert_allclose(
        cond,
        np.eye(2),
        rtol=1e-12,
        err_msg="a tiny but positive P(X=i) still defines P(Y | X=i): test p_x > 0, "
        "not a tolerance",
    )


def test_single_row_and_single_column_tables_are_valid():
    p_x, p_y, cond = _solve([[0.2, 0.3, 0.5]], "a single-row table must be accepted")
    np.testing.assert_allclose(p_x, [1.0], err_msg="one row: p_x must be [1.0]")
    np.testing.assert_allclose(p_y, [0.2, 0.3, 0.5], err_msg="one row: p_y must be the row itself")
    np.testing.assert_allclose(cond, [[0.2, 0.3, 0.5]], err_msg="one row: P(Y | X=0) is the row")
    p_x, p_y, cond = _solve([[0.2], [0.3], [0.5]], "a single-column table must be accepted")
    np.testing.assert_allclose(p_y, [1.0], err_msg="one column: p_y must be [1.0]")
    np.testing.assert_allclose(
        cond, [[1.0], [1.0], [1.0]], err_msg="one column: each P(Y | X=i) is certain"
    )


def test_integer_table_gives_float64_outputs():
    p_x, p_y, cond = _solve(np.array([[1, 0], [0, 0]]), "an integer table must be accepted")
    assert p_x.dtype == np.float64 and p_y.dtype == np.float64 and cond.dtype == np.float64, (
        "integer input must give float64 outputs, so conditionals are not truncated to integers"
    )
    np.testing.assert_allclose(p_x, [1.0, 0.0], err_msg="integer table: p_x wrong")
    np.testing.assert_allclose(cond[0], [1.0, 0.0], err_msg="integer table: P(Y | X=0) wrong")
    assert np.isnan(cond[1]).all(), "P(Y | X=1) is undefined because P(X=1) = 0"


def test_sum_within_rounding_of_one_is_accepted():
    _solve(
        np.full((3, 3), 1.0 / 9.0),
        "a table whose sum is 1 up to float rounding must be accepted: an exact sum == 1 check "
        "rejects rounding error, the tolerance is 1e-9",
    )
    # A sum 5e-10 above 1 is inside the 1e-9 tolerance, so it must be accepted, not rejected.
    p_x, _, _ = _solve(
        np.array([[0.5, 0.5 + 5e-10]]),
        "a sum within 1e-9 of 1 must be accepted: the tolerance is too strict",
    )
    np.testing.assert_allclose(
        p_x,
        [1.0 + 5e-10],
        rtol=1e-9,
        err_msg="a sum within 1e-9 of 1 is accepted, and P(X) is the row sum unchanged",
    )


def test_sum_outside_tolerance_is_rejected_on_both_sides():
    _expect_value_error(
        [[0.5 * (1 + 1e-6), 0.5 * (1 + 1e-6)]],
        "a table summing to 1 + 1e-6 must raise ValueError (outside the 1e-9 tolerance)",
    )
    _expect_value_error(
        [[0.5 * (1 - 1e-6), 0.5 * (1 - 1e-6)]],
        "a table summing to 1 - 1e-6 must raise ValueError (outside the 1e-9 tolerance)",
    )


def test_negative_entry_is_rejected_even_when_sum_is_one():
    _expect_value_error(
        [[1.5, -0.5], [0.0, 0.0]],
        "a negative probability must raise ValueError, even when the entries sum to 1",
    )


def test_nan_entry_is_rejected():
    _expect_value_error(
        [[0.5, np.nan], [0.0, 0.5]],
        "a NaN entry must raise ValueError: NaN fails every comparison, so a sign or sum "
        "check alone lets it through",
    )


def test_one_dimensional_input_is_rejected():
    _expect_value_error([0.5, 0.5], "a 1-D input is not a joint table: raise ValueError")


def test_empty_table_is_rejected():
    _expect_value_error(
        np.zeros((0, 0)), "an empty table has no probability mass: raise ValueError"
    )


def test_input_table_is_not_mutated():
    joint = np.array([[0.1, 0.2], [0.3, 0.4 + 5e-10]])
    before = joint.copy()
    _solve(joint, "a table with sum 1 + 5e-10 must be accepted")
    np.testing.assert_array_equal(
        joint, before, err_msg="input table was modified in place: normalise a copy instead"
    )


def test_non_2d_table_with_valid_probabilities_is_rejected():
    _expect_value_error(
        np.full((2, 2, 2), 0.125),
        "a 3-D array is not a joint table: raise ValueError even when the entries sum to 1",
    )


def test_sum_outside_the_1e9_tolerance_by_a_small_margin_is_rejected():
    _expect_value_error(
        [[0.5, 0.5 + 1.5e-9]],
        "a table whose sum is 1.5e-9 above 1 is outside the 1e-9 tolerance: raise ValueError",
    )
    _expect_value_error(
        [[0.5, 0.5 - 1.5e-9]],
        "a table whose sum is 1.5e-9 below 1 is outside the 1e-9 tolerance: raise ValueError",
    )
