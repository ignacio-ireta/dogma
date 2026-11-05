from submission import double_all


def test_doubles_each_value() -> None:
    assert double_all([1.0, 2.5]) == [2.0, 5.0]


def test_empty_input_gives_empty_output() -> None:
    assert double_all([]) == []


def test_negative_values_stay_negative() -> None:
    assert double_all([-3.0]) == [-6.0]
