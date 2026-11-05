from __future__ import annotations

import itertools
import math
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import pass_at_k


def checked(call):
    try:
        value = call()
    except Exception:
        pytest.fail("valid input raised: honor the documented contract", pytrace=False)
    if not isinstance(value, float):
        pytest.fail("return a plain float, not an exact Fraction, int or other type", pytrace=False)
    return value


def rejects(call, exc):
    try:
        call()
    except exc:
        return
    except Exception:
        pytest.fail(f"invalid input must raise {exc.__name__}", pytrace=False)
    pytest.fail(f"invalid input must raise {exc.__name__}", pytrace=False)


def exact(n: int, c: int, k: int) -> float:
    # Independent oracle: exact rational arithmetic on the binomial coefficients, then one rounding.
    return float(Fraction(1) - Fraction(math.comb(n - c, k), math.comb(n, k)))


def close(got: float, want: float, *, rel: float = 1e-9, abs_: float = 1e-12, detail: str) -> None:
    if not math.isclose(got, want, rel_tol=rel, abs_tol=abs_):
        pytest.fail(detail, pytrace=False)


@settings(max_examples=100, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.integers(1, 60), st.data())
def test_small_inputs_match_the_exact_binomial_ratio(n, data):
    c = data.draw(st.integers(0, n))
    k = data.draw(st.integers(1, n))
    got = checked(lambda: pass_at_k(n, c, k))
    close(
        got,
        exact(n, c, k),
        detail=(
            "value differs from the exact binomial ratio: check the k-subset formula and the "
            "treatment of c = 0 and n - c < k"
        ),
    )


def test_draws_are_without_replacement_not_the_with_replacement_formula():
    # 1 - (1 - c/n)^k assumes replacement. With n=10, c=3, k=5 the exact value is 1 - 21/252.
    got = checked(lambda: pass_at_k(10, 3, 5))
    close(
        got,
        1 - 21 / 252,
        detail=(
            "used the with-replacement formula 1 - (1 - c/n)^k; draws must be without replacement"
        ),
    )


def test_large_n_with_huge_binomials_stays_finite_and_exact():
    # C(2000, 1000) overflows a float, but the answer is exactly 1/2: C(1999, 1000) / C(2000, 1000).
    got = checked(lambda: pass_at_k(2000, 1, 1000))
    close(got, 0.5, detail="large n overflowed or lost accuracy: never build C(n, k) as a float")


def test_tiny_answers_keep_relative_accuracy():
    # With c = 1 the exact answer is k / n. Subtracting a product from 1 would lose all digits here.
    for n, k in [(10**6, 1), (10**6, 2), (10**9, 3), (10**12, 1)]:
        got = checked(lambda n=n, k=k: pass_at_k(n, 1, k))
        close(
            got,
            k / n,
            rel=1e-9,
            abs_=0.0,
            detail=(
                "tiny result lost its digits: 1 - product cancels; use log1p and expm1 on the sum "
                "of logs"
            ),
        )


def test_boundary_where_n_minus_c_equals_k_is_not_forced_to_one():
    # n - c = k: only the all-failing subset is wrong, so the answer is 1 - 1/C(5, 3) = 0.9.
    got = checked(lambda: pass_at_k(5, 2, 3))
    close(
        got,
        0.9,
        detail=(
            "boundary off by one: when n - c == k a failing subset still exists, so the answer is "
            "below 1"
        ),
    )


def test_all_samples_correct_and_full_budget_are_exact():
    assert checked(lambda: pass_at_k(7, 7, 4)) == 1.0, "c = n must give exactly 1.0"
    assert checked(lambda: pass_at_k(7, 0, 7)) == 0.0, "c = 0 must give exactly 0.0"
    assert checked(lambda: pass_at_k(7, 3, 7)) == 1.0, "k = n with any pass must give exactly 1.0"


@settings(max_examples=100, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.integers(2, 80), st.data())
def test_value_never_decreases_as_budget_grows(n, data):
    c = data.draw(st.integers(0, n))
    values = [checked(lambda k=k: pass_at_k(n, c, k)) for k in range(1, n + 1)]
    for lower, higher in itertools.pairwise(values):
        if higher + 1e-12 < lower:
            pytest.fail(
                "pass@k decreased as k grew: a larger budget cannot be worse", pytrace=False
            )


def test_integer_and_large_inputs_are_accepted():
    got = checked(lambda: pass_at_k(10**12, 10**12 - 1, 2))
    assert 0.0 < got <= 1.0, "huge integer inputs must be accepted and give a probability"


def test_non_int_arguments_raise_type_error():
    rejects(lambda: pass_at_k(5.0, 2, 1), TypeError)  # type: ignore[arg-type]
    rejects(lambda: pass_at_k(5, True, 1), TypeError)
    rejects(lambda: pass_at_k(5, 2, "1"), TypeError)  # type: ignore[arg-type]


def test_out_of_range_arguments_raise_value_error():
    rejects(lambda: pass_at_k(0, 0, 1), ValueError)
    rejects(lambda: pass_at_k(5, 6, 1), ValueError)
    rejects(lambda: pass_at_k(5, -1, 1), ValueError)
    rejects(lambda: pass_at_k(5, 2, 0), ValueError)
    rejects(lambda: pass_at_k(5, 2, 6), ValueError)
