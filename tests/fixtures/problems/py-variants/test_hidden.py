import pytest
from submission import add


def _raises_type_error(call, mistake):
    try:
        call()
    except TypeError:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; raised {type(exc).__name__} instead")
    pytest.fail(mistake)


def test_sum_of_negative_ints():
    assert add(-4, -6) == -10, "the sum is wrong for negative ints: check the sign"


def test_large_ints_stay_exact():
    assert add(10**30, 1) == 10**30 + 1, "large ints lost precision: do not convert to float"


def test_mixed_int_and_float():
    assert add(1, 0.5) == 1.5, "an int plus a float should be their sum"


def test_none_raises_type_error():
    _raises_type_error(lambda: add(None, 1), "None is not a number: raise TypeError")


def test_text_raises_type_error():
    _raises_type_error(lambda: add("a", "b"), "text is not a number: raise TypeError")
