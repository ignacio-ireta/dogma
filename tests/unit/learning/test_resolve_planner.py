"""Consolidate's re-solve block: due timed re-solves first, labelled untimed practice otherwise.

Table-driven over pure plan facts, like the rest of test_planner.py. Low-energy sessions never
become timed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from tests.unit.learning.test_planner import PER_DAY, facts, familiar, plan, topic

from dogma.db.models import Mastery
from dogma.learning.planner import Energy, PlanFacts, ResolveDue, SessionPlan

DUE_AT = datetime(2026, 10, 20, 9, tzinfo=UTC)


def due(
    topic_id: str,
    *,
    minutes: int = 20,
    stage: int = 0,
    open_window: bool = False,
    due_at: datetime = DUE_AT,
) -> ResolveDue:
    return ResolveDue(
        problem_id=f"{topic_id}/p01-core",
        topic_id=topic_id,
        title=f"{topic_id} core",
        minutes=minutes,
        stage=stage,
        due_at=due_at,
        open=open_window,
    )


def with_due(*items: ResolveDue, **kwargs: object) -> PlanFacts:
    base = facts(topic("tst/a"), topic("tst/b"), topic("tst/c"), topic("tst/d"), **kwargs)  # type: ignore[arg-type]
    return PlanFacts(
        topics=base.topics,
        mastery=base.mastery,
        due_count=base.due_count,
        familiar=base.familiar,
        resolves=tuple(items),
    )


def resolve_block(result: SessionPlan) -> str:
    (block,) = [b for b in result.blocks if b.name == "Re-solve"]
    return block.detail


def consolidate(world: PlanFacts, **kwargs: object) -> SessionPlan:
    return plan(world, slot=PER_DAY, **kwargs)


def test_a_due_timed_resolve_is_the_block_and_names_the_command_budget_and_no_hints() -> None:
    result = consolidate(with_due(due("tst/a", minutes=20)))

    assert result.blocks[1].minutes == 25
    detail = resolve_block(result)
    assert "tst/a/p01-core" in detail
    assert "budget 20 min" in detail
    assert "no hints" in detail
    assert result.first_action.startswith("Start the timed re-solve: dogma resolve tst/a/p01-core")
    assert "untimed" not in detail.lower()
    assert getattr(result, "timed_problem_id", None) == "tst/a/p01-core"


def test_lowest_stage_comes_first_then_the_earliest_due() -> None:
    later = datetime(2026, 10, 25, tzinfo=UTC)
    world = with_due(
        due("tst/a", stage=1, due_at=DUE_AT),
        due("tst/b", stage=0, due_at=later),
    )

    result = consolidate(world)

    assert "dogma resolve tst/b/p01-core" in result.first_action


def test_two_problems_whose_budgets_fit_the_block_are_both_chosen() -> None:
    world = with_due(due("tst/a", minutes=10), due("tst/b", minutes=10), due("tst/c", minutes=10))

    detail = resolve_block(consolidate(world))

    assert "tst/a/p01-core" in detail and "tst/b/p01-core" in detail
    assert "tst/c/p01-core" not in detail


def test_budgets_that_do_not_fit_together_pick_only_the_first_that_fits() -> None:
    world = with_due(due("tst/a", minutes=20), due("tst/b", minutes=20))

    detail = resolve_block(consolidate(world))

    assert "tst/a/p01-core" in detail
    assert "tst/b/p01-core" not in detail


def test_a_problem_over_the_block_is_not_timed_and_falls_back_to_untimed() -> None:
    result = consolidate(with_due(due("tst/a", minutes=30)))

    assert "Untimed" in resolve_block(result)
    assert getattr(result, "timed_problem_id", None) is None
    assert "dogma resolve" not in result.first_action


def test_todays_learn_topic_is_never_the_timed_re_solve() -> None:
    result = consolidate(
        with_due(due("tst/a")),
        exclude=frozenset({"tst/a"}),
    )

    assert "dogma resolve" not in result.first_action
    assert "Untimed" in result.blocks[1].detail


def test_with_nothing_due_the_familiar_re_solve_is_labelled_untimed() -> None:
    world = facts(topic("tst/a"), familiar=(familiar("tst/a"),))

    result = consolidate(world)

    assert "Untimed" in resolve_block(result)
    assert "dogma resolve" not in result.first_action


def test_an_open_window_is_finished_first_and_named_as_open() -> None:
    result = consolidate(with_due(due("tst/a", open_window=True)))

    assert result.first_action == (
        "Finish the open timed re-solve: dogma submit tst/a/p01-core, "
        "or dogma resolve tst/a/p01-core --abandon."
    )


@pytest.mark.parametrize("energy", [Energy.LOW])
def test_minimum_viable_sessions_stay_untimed_even_when_re_solves_are_due(energy: Energy) -> None:
    result = plan(with_due(due("tst/a")), slot=1, energy=energy)

    assert "dogma resolve" not in " ".join(block.detail for block in result.blocks)
    assert "dogma resolve" not in result.first_action
    assert "Untimed" in resolve_block(result)


def test_no_plan_text_counts_a_backlog() -> None:
    world = with_due(due("tst/a"), due("tst/b", stage=2), due("tst/c", stage=3), mastery={})

    texts = []
    for slot in (1, PER_DAY):
        result = plan(world, slot=slot)
        texts.append(result.first_action)
        texts.extend(block.detail for block in result.blocks)
    joined = " ".join(texts).lower()

    for word in ("overdue", "backlog", "debt", "streak"):
        assert word not in joined


def test_mastery_argument_is_accepted_for_planner_facts() -> None:
    world = with_due(due("tst/a"), mastery={"tst/a": Mastery.RETAINED})

    assert "dogma resolve" in consolidate(world).first_action
