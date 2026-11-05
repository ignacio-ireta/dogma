from submission import add


def test_adds_two_positive_integers():
    assert add(2, 3) == 5, "add(2, 3) should be 5"


def test_adds_two_negative_integers():
    assert add(-4, -6) == -10, "the sum of two negatives should be negative"


def test_adding_zero_is_the_identity():
    assert add(7, 0) == 7, "adding zero should leave the value unchanged"
