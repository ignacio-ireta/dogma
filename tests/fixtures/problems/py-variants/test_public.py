from submission import add


def test_example_adds_two_ints():
    assert add(2, 3) == 5, "add(2, 3) should be 5"


def test_example_adds_two_floats():
    assert add(0.5, 0.25) == 0.75, "add(0.5, 0.25) should be 0.75"


def test_adding_zero_changes_nothing():
    assert add(7, 0) == 7, "adding zero should leave the value unchanged"
