"""The planner is pure: table-driven budgets, prerequisite gating, exclusions and empty states."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dogma.db.models import Mastery, mastery_rank
from dogma.learning.planner import (
    CONSOLIDATE_MINUTES,
    LEARN_MINUTES,
    MINIMUM_MINUTES,
    Energy,
    FamiliarProblem,
    PlanFacts,
    ProblemFacts,
    SessionKind,
    SessionPlan,
    TopicFacts,
    plan_session,
    session_kind,
)

PER_DAY = 3


def problem(topic_id: str, status: str = "reviewed", *, core: bool = True) -> ProblemFacts:
    return ProblemFacts(
        problem_id=f"{topic_id}/p01-core", title=f"{topic_id} core", status=status, core=core
    )


def topic(
    topic_id: str,
    *,
    status: str | None = "reviewed",
    requires: tuple[str, ...] = (),
    core_status: str = "reviewed",
) -> TopicFacts:
    return TopicFacts(
        topic_id=topic_id,
        title=topic_id,
        status=status,
        objective=f"Explain {topic_id} in one sentence",
        requires=requires,
        problems=(problem(topic_id, core_status),),
    )


def facts(
    *topics: TopicFacts,
    mastery: dict[str, Mastery] | None = None,
    due: int = 0,
    familiar: tuple[FamiliarProblem, ...] = (),
) -> PlanFacts:
    return PlanFacts(topics=tuple(topics), mastery=mastery or {}, due_count=due, familiar=familiar)


def plan(world: PlanFacts, slot: int = 1, **kwargs: object) -> SessionPlan:
    energy = kwargs.pop("energy", Energy.NORMAL)
    exclude = kwargs.pop("exclude", frozenset())
    assert isinstance(energy, Energy) and isinstance(exclude, frozenset)
    return plan_session(world, slot=slot, sessions_per_day=PER_DAY, energy=energy, exclude=exclude)


def familiar(topic_id: str) -> FamiliarProblem:
    return FamiliarProblem(problem_id=f"{topic_id}/p01-core", topic_id=topic_id, title=topic_id)


def blocks(plan_: SessionPlan) -> list[tuple[str, int]]:
    return [(block.name, block.minutes) for block in plan_.blocks]


# --- budgets --------------------------------------------------------------------------------


def test_learn_session_is_the_documented_five_block_budget() -> None:
    result = plan(facts(topic("tst/a")), slot=1)

    assert result.kind is SessionKind.LEARN
    assert blocks(result) == [
        ("Retrieve", 5),
        ("Predict", 3),
        ("Learn", 12),
        ("Practice", 20),
        ("Close", 5),
    ]
    assert result.minutes == LEARN_MINUTES == 45


def test_consolidate_session_is_review_resolve_close() -> None:
    result = plan(facts(topic("tst/a")), slot=PER_DAY)

    assert result.kind is SessionKind.CONSOLIDATE
    assert blocks(result) == [("Review", 15), ("Re-solve", 25), ("Close", 5)]
    assert result.minutes == CONSOLIDATE_MINUTES == 45


def test_low_energy_session_is_review_and_familiar_resolve_only() -> None:
    result = plan(
        facts(topic("tst/a"), due=30, familiar=(familiar("tst/a"),)),
        slot=1,
        energy=Energy.LOW,
    )

    assert result.kind is SessionKind.MINIMUM
    assert result.energy is Energy.LOW
    assert blocks(result) == [("Review", 10), ("Re-solve", 10)]
    assert result.minutes == MINIMUM_MINUTES == 20


@pytest.mark.parametrize("slot", [0, PER_DAY + 1])
def test_a_slot_outside_the_day_is_rejected(slot: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        session_kind(slot, PER_DAY)


def test_only_the_last_slot_consolidates() -> None:
    assert [session_kind(slot, PER_DAY) for slot in range(1, PER_DAY + 1)] == [
        SessionKind.LEARN,
        SessionKind.LEARN,
        SessionKind.CONSOLIDATE,
    ]


# --- prerequisites and content readiness ----------------------------------------------------


def test_a_topic_waits_until_its_prerequisite_is_solved() -> None:
    waiting = facts(topic("tst/a"), topic("tst/b", requires=("tst/a",)))
    assert plan(waiting).topic_id == "tst/a"

    solved = facts(
        topic("tst/a"),
        topic("tst/b", requires=("tst/a",)),
        mastery={"tst/a": Mastery.SOLVED},
    )
    assert plan(solved).topic_id == "tst/b"


@pytest.mark.parametrize("state", [Mastery.INTRODUCED, None])
def test_planned_or_introduced_prerequisites_never_count_as_solved(state: Mastery | None) -> None:
    mastery = {} if state is None else {"tst/a": state}
    world = facts(topic("tst/a"), topic("tst/b", requires=("tst/a",)), mastery=mastery)

    result = plan(world, slot=2, exclude=frozenset({"tst/a"}))

    assert result.topic_id is None
    assert result.kind is SessionKind.CONSOLIDATE
    assert result.notes == (
        "Nothing new is ready: tst/b needs tst/a solved first. This session reviews instead.",
    )


def test_a_prerequisite_solved_outside_the_path_still_counts() -> None:
    world = facts(
        topic("tst/b", requires=("elsewhere/x",)),
        mastery={"elsewhere/x": Mastery.SOLVED},
    )

    assert plan(world).topic_id == "tst/b"


def test_a_prerequisite_with_no_content_and_no_solve_blocks() -> None:
    result = plan(facts(topic("tst/b", requires=("missing/x",))))

    assert result.topic_id is None
    assert "tst/b needs missing/x solved first" in result.notes[0]


@pytest.mark.parametrize(
    "unavailable",
    [
        topic("tst/a", status="draft"),
        topic("tst/a", status="retired"),
        topic("tst/a", status=None),
        topic("tst/a", core_status="draft"),
    ],
    ids=["draft", "retired", "missing-package", "draft-core-problem"],
)
def test_draft_retired_missing_or_unready_content_is_never_planned(unavailable: TopicFacts) -> None:
    result = plan(facts(unavailable))

    assert result.topic_id is None and result.must_do is None
    assert result.notes == (
        "Nothing new is ready: tst/a has no verified or reviewed content yet. "
        "This session reviews instead.",
    )


def test_verified_content_is_ready_like_reviewed_content() -> None:
    assert plan(facts(topic("tst/a", status="verified", core_status="verified"))).topic_id == (
        "tst/a"
    )


def test_a_problem_that_is_not_core_does_not_make_a_topic_ready() -> None:
    not_core = TopicFacts(
        topic_id="tst/a",
        title="tst/a",
        status="reviewed",
        objective="x",
        requires=(),
        problems=(problem("tst/a", core=False),),
    )

    assert not_core.ready_core() is None


def test_the_must_do_is_the_topics_core_problem() -> None:
    result = plan(facts(topic("tst/a")))

    assert result.must_do == "tst/a/p01-core"
    assert "dogma open tst/a/p01-core" in result.blocks[3].detail


# --- which topic a Learn session takes ------------------------------------------------------


def test_unfinished_topics_come_before_new_ones_then_path_order() -> None:
    world = facts(
        topic("tst/a"),
        topic("tst/b"),
        topic("tst/c"),
        mastery={"tst/c": Mastery.INTRODUCED},
    )

    assert plan(world).topic_id == "tst/c"


def test_topics_planned_earlier_today_are_skipped() -> None:
    result = plan(facts(topic("tst/a"), topic("tst/b")), slot=2, exclude=frozenset({"tst/a"}))

    assert result.topic_id == "tst/b"


def test_a_learn_slot_with_nothing_new_becomes_a_review_with_the_reason() -> None:
    result = plan(facts(topic("tst/a"), mastery={"tst/a": Mastery.SOLVED}))

    assert result.kind is SessionKind.CONSOLIDATE
    assert result.topic_id is None
    assert result.notes == (
        "Nothing new is ready: every topic on the core path is already solved. "
        "This session reviews instead.",
    )


def test_a_slot_whose_only_topics_are_planned_today_says_so() -> None:
    result = plan(facts(topic("tst/a")), exclude=frozenset({"tst/a"}))

    assert result.notes == (
        "Nothing new is ready: the remaining topics are already planned for today. "
        "This session reviews instead.",
    )


# --- low energy ------------------------------------------------------------------------------


def test_low_energy_never_introduces_new_material_even_when_topics_are_ready() -> None:
    result = plan(
        facts(topic("tst/a"), topic("tst/b"), due=2, familiar=(familiar("tst/z"),)),
        energy=Energy.LOW,
    )

    assert result.topic_id is None
    assert result.must_do is None
    assert "lesson.md" not in " ".join(block.detail for block in result.blocks)


def test_low_energy_empty_state_says_what_to_do_and_plans_nothing_new() -> None:
    result = plan(facts(topic("tst/a")), energy=Energy.LOW)

    assert result.first_action == "Finish with dogma close, then rest. Nothing else is planned."
    assert result.notes == (
        "Nothing is due and no solved problem is ready to re-solve. Low-energy fallback: "
        "finish with dogma close, rest, and come back. No new material is planned.",
    )
    assert result.minutes == MINIMUM_MINUTES


def test_low_energy_with_nothing_due_re_solves_the_oldest_familiar_problem() -> None:
    result = plan(facts(topic("tst/a"), familiar=(familiar("tst/a"),)), energy=Energy.LOW)

    assert result.first_action == "Open tst/a/p01-core and read example 1."
    assert result.notes == ()


# --- due cards and familiar problems --------------------------------------------------------


@pytest.mark.parametrize(
    ("due", "expected"),
    [
        (40, "Answer up to 15 due cards (40 due now): dogma review --limit 15"),
        (3, "Answer up to 3 due cards (3 due now): dogma review --limit 3"),
    ],
)
def test_the_consolidate_review_is_capped_by_block_and_queue(due: int, expected: str) -> None:
    result = plan(facts(topic("tst/a"), due=due), slot=PER_DAY)

    assert result.blocks[0].detail == expected
    assert result.first_action == expected.split(": ", 1)[1]


def test_retrieve_takes_five_cards_at_most() -> None:
    result = plan(facts(topic("tst/a"), due=12), slot=1)

    assert (
        result.blocks[0].detail == "Answer up to 5 due cards (12 due now): dogma review --limit 5"
    )
    assert result.first_action == "dogma review --limit 5"


def test_an_empty_review_queue_is_stated_plainly() -> None:
    result = plan(facts(topic("tst/a")), slot=PER_DAY)

    assert result.blocks[0].detail.startswith("No cards are due right now.")


def test_re_solve_picks_the_first_familiar_problem_outside_excluded_topics() -> None:
    world = facts(topic("tst/a"), familiar=(familiar("tst/a"), familiar("tst/b")))

    result = plan(world, slot=PER_DAY, exclude=frozenset({"tst/a"}))

    assert "tst/b/p01-core" in result.blocks[1].detail
    assert result.first_action == "Open tst/b/p01-core and read example 1."


def test_re_solve_says_so_when_nothing_is_solved_yet() -> None:
    result = plan(facts(topic("tst/a")), slot=PER_DAY)

    assert result.blocks[1].detail.startswith("Nothing solved is ready to re-solve yet.")


def test_the_same_facts_always_give_the_same_plan() -> None:
    world = facts(
        topic("tst/a"),
        topic("tst/b", requires=("tst/a",)),
        due=4,
        familiar=(familiar("tst/z"),),
    )

    assert plan(world) == plan(world)


# --- invariants over random facts -----------------------------------------------------------

_TOPIC_IDS = ("tst/a", "tst/b", "tst/c", "tst/d")
_TOPICS = (
    topic("tst/a"),
    topic("tst/b", requires=("tst/a",)),
    topic("tst/c", status="draft"),
    topic("tst/d", requires=("tst/b",)),
)


@settings(max_examples=200, deadline=None)
@given(
    states=st.lists(st.sampled_from([None, *Mastery]), min_size=4, max_size=4),
    due=st.integers(min_value=0, max_value=60),
    slot=st.integers(min_value=1, max_value=PER_DAY),
    low=st.booleans(),
    excluded=st.sets(st.sampled_from(_TOPIC_IDS)),
    familiar_ids=st.lists(st.sampled_from(_TOPIC_IDS), unique=True),
)
def test_invariants_hold_for_any_facts(
    *,
    states: list[Mastery | None],
    due: int,
    slot: int,
    low: bool,
    excluded: set[str],
    familiar_ids: list[str],
) -> None:
    mastery = {topic_id: state for topic_id, state in zip(_TOPIC_IDS, states, strict=True) if state}
    world = PlanFacts(
        topics=_TOPICS,
        mastery=mastery,
        due_count=due,
        familiar=tuple(familiar(topic_id) for topic_id in familiar_ids),
    )

    result = plan(
        world,
        slot=slot,
        energy=Energy.LOW if low else Energy.NORMAL,
        exclude=frozenset(excluded),
    )

    assert result.first_action
    if low:
        assert result.minutes == MINIMUM_MINUTES
        assert result.topic_id is None and result.must_do is None
    elif result.kind is SessionKind.LEARN:
        assert result.minutes == LEARN_MINUTES
        assert result.topic_id is not None and result.topic_id not in excluded
        chosen = next(item for item in _TOPICS if item.topic_id == result.topic_id)
        assert chosen.ready_core() is not None
        assert mastery_rank(mastery.get(result.topic_id, Mastery.UNSEEN)) < mastery_rank(
            Mastery.SOLVED
        )
        for requirement in chosen.requires:
            assert mastery_rank(mastery.get(requirement, Mastery.UNSEEN)) >= mastery_rank(
                Mastery.SOLVED
            )
    else:
        assert result.minutes == CONSOLIDATE_MINUTES
        assert result.topic_id is None
