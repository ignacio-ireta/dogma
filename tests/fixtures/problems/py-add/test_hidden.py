import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import add

SMALL_INTS = st.integers(min_value=-(10**6), max_value=10**6)


@settings(max_examples=50, deadline=None)
@given(SMALL_INTS, SMALL_INTS)
def test_matches_python_addition_on_random_integers(a, b):
    assert add(a, b) == a + b, f"add({a}, {b}) should equal {a + b}"


@settings(max_examples=50, deadline=None)
@given(SMALL_INTS, SMALL_INTS)
def test_result_does_not_depend_on_argument_order(a, b):
    assert add(a, b) == add(b, a), "addition should be commutative"


def test_large_integers_are_exact():
    assert add(10**30, 1) == 10**30 + 1, "integers beyond float precision should stay exact"


def test_floats_are_summed():
    assert add(0.5, 0.25) == pytest.approx(0.75), "floats should add with normal precision"


def test_none_raises_type_error():
    with pytest.raises(TypeError):
        add(None, 1)
