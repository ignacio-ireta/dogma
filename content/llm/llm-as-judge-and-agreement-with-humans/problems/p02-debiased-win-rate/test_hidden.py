from __future__ import annotations

import functools
import itertools
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import debiased_win_rate

from dogma_data import load_frame

VERDICTS = ["A", "B", "tie"]

# Independent oracle: a lookup table over all nine (forward, reverse) combinations, written as the
# candidate's score directly. "A" forward and "B" reverse both mean X wins.
TABLE = {}
for forward, reverse in itertools.product(VERDICTS, VERDICTS):
    f = {"A": 1, "B": 0, "tie": None}[forward]  # 1 = X wins forward
    r = {"B": 1, "A": 0, "tie": None}[reverse]  # 1 = X wins reverse
    if f == 1 and r == 1:
        TABLE[(forward, reverse)] = 1.0
    elif f == 0 and r == 0:
        TABLE[(forward, reverse)] = 0.0
    else:
        TABLE[(forward, reverse)] = 0.5


def checked(call):
    try:
        value = call()
    except Exception:
        pytest.fail("valid input raised: honor the documented contract", pytrace=False)
    if not isinstance(value, float):
        pytest.fail("return the win rate as a plain float", pytrace=False)
    return value


def rejects(call, exc):
    try:
        call()
    except exc:
        return
    except Exception:
        pytest.fail(f"invalid input must raise {exc.__name__}", pytrace=False)
    pytest.fail(f"invalid input must raise {exc.__name__}", pytrace=False)


@functools.cache
def _human_judgments():
    """Real verdict pairs: MT-Bench items that human experts judged in both orders.

    An item is a question, a turn and two chat models; `X` is the model whose name sorts
    first. `forward` is the first recorded verdict with `X` shown as model A and `reverse` the
    first with `Y` shown as model A, each read as "A", "B" or "tie" for the position that won.
    The two verdicts come from different experts, so they disagree more often than one
    judge's position bias alone would make them. 582 items.
    """
    table = load_frame("mt_bench_judgments")
    seen: dict[tuple[int, int, str, str], dict[bool, str]] = {}
    for row in table.itertuples():
        first, second = sorted((row.model_a, row.model_b))
        verdict = {"model_a": "A", "model_b": "B", "tie": "tie"}[row.winner]
        orders = seen.setdefault((int(row.question_id), int(row.turn), first, second), {})
        orders.setdefault(row.model_a == first, verdict)
    return tuple((orders[True], orders[False]) for orders in seen.values() if len(orders) == 2)


@settings(max_examples=100, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(st.lists(st.integers(min_value=0, max_value=581), min_size=1, max_size=30))
def test_random_judgments_match_the_lookup_table_oracle(judgments):
    # Each number picks a real item judged by human experts in both orders. All nine
    # combinations of verdicts occur among them, the rarest 22 times.
    real = _human_judgments()
    judgments = [real[j % len(real)] for j in judgments]
    got = checked(lambda: debiased_win_rate(judgments))
    want = sum(TABLE[j] for j in judgments) / len(judgments)
    if not math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12):
        pytest.fail(
            "win rate differs from the nine-case table: check which position each verdict names "
            "in the reverse order, and that inconsistent pairs score 0.5",
            pytrace=False,
        )


@pytest.mark.parametrize("verdict", ["A", "B"], ids=["verdict-a", "verdict-b"])
def test_constant_position_verdicts_give_one_half_in_either_direction(verdict):
    got = checked(lambda: debiased_win_rate([(verdict, verdict)] * 7))
    assert math.isclose(got, 0.5, abs_tol=1e-12), (
        "always the same position in both orders is position bias: it must give 0.5"
    )


def test_consistent_judge_favouring_x_gives_one():
    got = checked(lambda: debiased_win_rate([("A", "B")] * 5))
    assert got == 1.0, "X wins in both orders for every item: the rate must be exactly 1.0"


def test_consistent_judge_favouring_y_gives_zero():
    got = checked(lambda: debiased_win_rate([("B", "A")] * 5))
    assert got == 0.0, "Y wins in both orders for every item: the rate must be exactly 0.0"


def test_tie_and_one_decisive_verdict_do_not_count_as_a_win():
    got = checked(lambda: debiased_win_rate([("A", "tie"), ("tie", "B")]))
    assert math.isclose(got, 0.5, abs_tol=1e-12), (
        "one tie with one decisive verdict is not a consistent win: each item scores 0.5"
    )


def test_all_ties_give_one_half():
    got = checked(lambda: debiased_win_rate([("tie", "tie")] * 4))
    assert math.isclose(got, 0.5, abs_tol=1e-12), "ties score one half each"


def test_order_of_items_does_not_change_the_rate():
    items = [(VERDICTS[i % 3], VERDICTS[(i // 3) % 3]) for i in range(25)]
    base = checked(lambda: debiased_win_rate(items))
    for shift in (1, 7, 13):
        rotated = items[shift:] + items[:shift]
        got = checked(lambda rotated=rotated: debiased_win_rate(rotated))
        assert math.isclose(got, base, abs_tol=1e-12), (
            "the win rate is a mean over items, so item order must not change it"
        )


def test_tuples_and_lists_are_both_accepted():
    got = checked(lambda: debiased_win_rate([["A", "B"], ("B", "A")]))
    assert math.isclose(got, 0.5, abs_tol=1e-12), (
        "pairs given as lists must be read the same way as tuples"
    )


def test_input_is_not_mutated():
    judgments = [("A", "B"), ("tie", "A")]
    before = list(judgments)
    checked(lambda: debiased_win_rate(judgments))
    assert judgments == before, "the judgment list must not be changed"


def test_empty_input_is_rejected():
    rejects(lambda: debiased_win_rate([]), ValueError)


def test_unknown_verdict_is_rejected():
    rejects(lambda: debiased_win_rate([("X", "B")]), ValueError)
    rejects(lambda: debiased_win_rate([("a", "B")]), ValueError)


def test_malformed_element_is_rejected():
    rejects(lambda: debiased_win_rate(["AB"]), TypeError)  # type: ignore[list-item]
    rejects(lambda: debiased_win_rate([("A", "B", "tie")]), TypeError)  # type: ignore[list-item]


def test_non_string_verdict_is_rejected():
    rejects(lambda: debiased_win_rate([(1, "B")]), TypeError)  # type: ignore[list-item]


def test_tuple_of_judgments_is_read_in_full():
    # Item scores: ("A", "B") is X in both orders (1.0), ("A", "A") is inconsistent (0.5),
    # ("B", "A") is Y in both orders (0.0). Mean = (1.0 + 0.5 + 0.0) / 3 = 0.5.
    got = checked(lambda: debiased_win_rate((("A", "B"), ("A", "A"), ("B", "A"))))
    assert math.isclose(got, 0.5, abs_tol=1e-12), (
        "a tuple of judgments must count every pair, the first one included"
    )
