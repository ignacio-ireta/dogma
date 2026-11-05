"""The session planner: a pure function from facts to one session (ARCHITECTURE §4.4, §5).

Nothing here reads a clock, a file or the database. ``dogma.learning.sessions`` gathers the facts
and stores the result. Equal facts always give an equal plan, so each rule is table-driven.

Rules:
- Only topics whose content is verified or reviewed, with a verified or reviewed core problem, are
  studied. Draft, retired and missing content is never planned.
- A new topic needs every ``requires`` topic at ``solved`` or further. A planned or introduced
  prerequisite does not count: only a real solve does.
- A Learn session finishes an introduced topic before it starts a new one, then takes the first
  eligible topic in path order. A topic already planned earlier the same day is skipped.
- Consolidate and low-energy sessions never introduce material. They draw on due cards and on one
  familiar problem: the one whose last passing submit is furthest back.
- Blocks always add up to the session's budget (45 minutes; 20 for low energy).
- A Consolidate session's 25-minute re-solve block is timed when a core problem's timed re-solve is
  due: lowest stage first, then earliest due, never a topic already planned today, at most two
  whose budgets fit the block. Otherwise it is familiar practice, labelled untimed. Minimum-viable
  sessions are always untimed.
- Interview sessions (I-18c): from path week 3, the first Consolidate slot of each week becomes an
  Interview session (a 40-minute mock, then a 5-minute debrief). Every session of the final path
  week is an Interview session. Low-energy sessions never are. The weakest part over the last three
  scored mocks is named in an Interview session's notes; no trend, streak or backlog is shown.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Final

from dogma.db.models import Mastery, mastery_rank

READY_STATUSES: Final = frozenset({"verified", "reviewed"})
LEARN_MINUTES: Final = 45
CONSOLIDATE_MINUTES: Final = 45
MINIMUM_MINUTES: Final = 20
RESOLVE_MINUTES: Final = 25  # Consolidate's timed re-solve block
MAX_TIMED_RESOLVES: Final = 2
INTERVIEW_FIRST_WEEK: Final = 3  # the first path week with one Interview session a week
INTERVIEW_FINAL_WEEK: Final = 14  # the interview week: every session of it is an Interview
MOCK_MINUTES: Final = 40
DEBRIEF_MINUTES: Final = 5
_CLOSE_DETAIL: Final = (
    "dogma close: where you stopped, the exact next step, any blocker, and a 60-second answer "
    "in your own words."
)


class Energy(StrEnum):
    NORMAL = "normal"
    LOW = "low"


class SessionKind(StrEnum):
    LEARN = "learn"
    CONSOLIDATE = "consolidate"
    MINIMUM = "minimum"
    INTERVIEW = "interview"


@dataclass(frozen=True)
class ProblemFacts:
    """One problem package as the planner sees it. ``status`` is its ``meta.yaml`` status.

    ``minutes`` is the problem's time budget, which a timed re-solve runs against.
    """

    problem_id: str
    title: str
    status: str
    core: bool
    minutes: int = 0


@dataclass(frozen=True)
class TopicFacts:
    """One topic on the core path. ``status`` is None when no usable package exists on disk."""

    topic_id: str
    title: str
    status: str | None
    objective: str
    requires: tuple[str, ...]
    problems: tuple[ProblemFacts, ...]

    def ready_core(self) -> ProblemFacts | None:
        """The core problem a learner can solve, or None when this topic cannot be studied."""
        if self.status not in READY_STATUSES:
            return None
        for problem in self.problems:
            if problem.core and problem.status in READY_STATUSES:
                return problem
        return None


@dataclass(frozen=True)
class FamiliarProblem:
    """A ready problem the learner has passed before. Callers list them oldest pass first."""

    problem_id: str
    topic_id: str
    title: str


@dataclass(frozen=True)
class ResolveDue:
    """A core problem whose timed re-solve is due now, or whose window is open. Stage 0 to 3."""

    problem_id: str
    topic_id: str
    title: str
    minutes: int
    stage: int
    due_at: datetime
    open: bool = False


@dataclass(frozen=True)
class PlanFacts:
    """Everything a plan may depend on. ``mastery`` lists stored states; missing means unseen.

    ``path_week`` is the core-path week of the local day (None outside the path). Interview
    sessions depend on it and on ``interviews_this_week`` (already planned this week).
    ``weak_part`` and ``weak_part_percent`` name the weakest part over the last scored mocks.
    """

    topics: tuple[TopicFacts, ...]
    mastery: Mapping[str, Mastery]
    due_count: int
    familiar: tuple[FamiliarProblem, ...]
    resolves: tuple[ResolveDue, ...] = ()
    path_week: int | None = None
    interviews_this_week: int = 0
    weak_part: str | None = None
    weak_part_percent: int | None = None


@dataclass(frozen=True)
class Block:
    name: str
    minutes: int
    detail: str


@dataclass(frozen=True)
class SessionPlan:
    kind: SessionKind
    energy: Energy
    blocks: tuple[Block, ...]
    topic_id: str | None
    must_do: str | None
    first_action: str
    notes: tuple[str, ...] = ()
    timed_problem_id: str | None = None

    @property
    def minutes(self) -> int:
        return sum(block.minutes for block in self.blocks)


def session_kind(slot: int, sessions_per_day: int) -> SessionKind:
    """The last session of a normal day consolidates; earlier ones learn (LEARNER_PROFILE §6)."""
    if not 1 <= slot <= sessions_per_day:
        raise ValueError(f"slot {slot} is outside the {sessions_per_day} sessions of a day")
    return SessionKind.CONSOLIDATE if slot == sessions_per_day else SessionKind.LEARN


def plan_session(
    facts: PlanFacts,
    *,
    slot: int,
    sessions_per_day: int,
    energy: Energy = Energy.NORMAL,
    exclude: frozenset[str] = frozenset(),
) -> SessionPlan:
    """The plan for one slot. ``exclude`` lists topics already planned for this local day."""
    if energy is Energy.LOW:
        return _minimum(facts, exclude)
    kind = session_kind(slot, sessions_per_day)
    if _is_interview(facts, kind):
        return _interview(facts)
    if kind is SessionKind.CONSOLIDATE:
        return _consolidate(facts, exclude)
    return _learn(facts, exclude)


def _is_interview(facts: PlanFacts, kind: SessionKind) -> bool:
    """Every session of the final week; otherwise the week's first Consolidate from week 3."""
    if facts.path_week is None:
        return False
    if facts.path_week == INTERVIEW_FINAL_WEEK:
        return True
    return (
        facts.path_week >= INTERVIEW_FIRST_WEEK
        and kind is SessionKind.CONSOLIDATE
        and facts.interviews_this_week == 0
    )


def _interview(facts: PlanFacts) -> SessionPlan:
    blocks = (
        Block(
            "Mock",
            MOCK_MINUTES,
            "Timed: a live coding problem, a theory question and a case, scored on the "
            "30-item mock rubric. Start with dogma mock start, then score with dogma mock score.",
        ),
        Block(
            "Debrief",
            DEBRIEF_MINUTES,
            "dogma mock debrief shows your part totals, the band and the case's debrief. Then "
            "dogma close.",
        ),
    )
    return SessionPlan(
        kind=SessionKind.INTERVIEW,
        energy=Energy.NORMAL,
        blocks=blocks,
        topic_id=None,
        must_do=None,
        first_action="Start the mock: dogma mock start (about 40 minutes, then the debrief).",
        notes=(_weak_part_note(facts),),
    )


def _weak_part_note(facts: PlanFacts) -> str:
    if facts.weak_part is None or facts.weak_part_percent is None:
        return "No scored mock yet. The weakest part is named here after your first score."
    return (
        f"Weakest part over your last three scored mocks: {facts.weak_part} "
        f"({facts.weak_part_percent}%)."
    )


def _learn(facts: PlanFacts, exclude: frozenset[str]) -> SessionPlan:
    choice, reason = _next_topic(facts, exclude)
    if choice is None:
        # Never invent material: an empty Learn slot becomes a review slot, with the reason shown.
        substitute = _consolidate(facts, exclude)
        return replace(
            substitute,
            notes=(f"Nothing new is ready: {reason}. This session reviews instead.",),
        )
    topic, core = choice
    lesson = f"content/{topic.topic_id}/lesson.md"
    objective = topic.objective
    due = facts.due_count
    blocks = (
        Block("Retrieve", 5, _review_detail(due, 5)),
        Block(
            "Predict",
            3,
            f"Before you read {lesson}, write one guess for this goal: {objective}",
        ),
        Block("Learn", 12, f"Read {lesson} (the system, the worked example, the interview angle)."),
        Block(
            "Practice",
            20,
            f"Must-do: dogma open {core.problem_id}, read example 1, then solve it. "
            f"Check public tests with dogma run {core.problem_id}, then "
            f"dogma submit {core.problem_id} to record a successful solve.",
        ),
        Block("Close", 5, _CLOSE_DETAIL),
    )
    if due:
        first = _review_command(due, 5)
    else:
        first = f"Write your Predict guess (3 min), then open {lesson}."
    return SessionPlan(
        kind=SessionKind.LEARN,
        energy=Energy.NORMAL,
        blocks=blocks,
        topic_id=topic.topic_id,
        must_do=core.problem_id,
        first_action=first,
    )


def _consolidate(facts: PlanFacts, exclude: frozenset[str]) -> SessionPlan:
    familiar = _familiar_choice(facts, exclude)
    timed = _timed_choice(facts.resolves, exclude)
    review, untimed = _review_and_resolve(facts.due_count, familiar, 15, RESOLVE_MINUTES)
    if timed:
        resolve = Block("Re-solve", RESOLVE_MINUTES, _timed_detail(timed))
        first = _timed_action(timed[0])
    else:
        resolve = untimed
        first = _review_or_resolve_action(facts.due_count, familiar, 15)
    blocks = (review, resolve, Block("Close", 5, _CLOSE_DETAIL))
    return SessionPlan(
        kind=SessionKind.CONSOLIDATE,
        energy=Energy.NORMAL,
        blocks=blocks,
        topic_id=None,
        must_do=None,
        first_action=first,
        timed_problem_id=timed[0].problem_id if timed else None,
    )


def _minimum(facts: PlanFacts, exclude: frozenset[str]) -> SessionPlan:
    familiar = _familiar_choice(facts, exclude)
    review, resolve = _review_and_resolve(facts.due_count, familiar, 10, 10)
    if facts.due_count == 0 and familiar is None:
        # Empty state: say so plainly, and plan nothing new.
        return SessionPlan(
            kind=SessionKind.MINIMUM,
            energy=Energy.LOW,
            blocks=(review, resolve),
            topic_id=None,
            must_do=None,
            first_action="Finish with dogma close, then rest. Nothing else is planned.",
            notes=(
                "Nothing is due and no solved problem is ready to re-solve. Low-energy fallback: "
                "finish with dogma close, rest, and come back. No new material is planned.",
            ),
        )
    return SessionPlan(
        kind=SessionKind.MINIMUM,
        energy=Energy.LOW,
        blocks=(review, resolve),
        topic_id=None,
        must_do=None,
        first_action=_review_or_resolve_action(facts.due_count, familiar, 10),
    )


def _review_and_resolve(
    due: int, familiar: FamiliarProblem | None, review_minutes: int, resolve_minutes: int
) -> tuple[Block, Block]:
    """The review block and an untimed familiar re-solve. Both are labelled untimed."""
    review = Block("Review", review_minutes, _review_detail(due, review_minutes))
    if familiar is None:
        resolve_detail = (
            "Nothing solved is ready to re-solve yet. Untimed: use the time on the review queue, "
            "or finish early with dogma close."
        )
    else:
        resolve_detail = (
            f"Untimed re-solve of {familiar.title} ({familiar.problem_id}), without the old "
            f"answer: dogma open {familiar.problem_id}, then dogma run {familiar.problem_id}."
        )
    return review, Block("Re-solve", resolve_minutes, resolve_detail)


def _timed_choice(
    resolves: tuple[ResolveDue, ...], exclude: frozenset[str]
) -> tuple[ResolveDue, ...]:
    """Open windows first (they are already running), then lowest stage, then earliest due.

    At most two items, and their budgets must fit the block. An open window is kept whatever its
    budget, because it is already running.
    """
    ordered = sorted(
        resolves, key=lambda item: (not item.open, item.stage, item.due_at, item.problem_id)
    )
    chosen: list[ResolveDue] = []
    minutes = 0
    for item in ordered:
        if item.topic_id in exclude or item.minutes <= 0 or len(chosen) == MAX_TIMED_RESOLVES:
            continue
        if not item.open and minutes + item.minutes > RESOLVE_MINUTES:
            continue
        chosen.append(item)
        minutes += item.minutes
    return tuple(chosen)


def _timed_detail(chosen: tuple[ResolveDue, ...]) -> str:
    parts = [f"{item.title} ({item.problem_id}), budget {item.minutes} min" for item in chosen]
    first = chosen[0]
    if first.open:
        tail = (
            f" The open one is running: dogma submit {first.problem_id} when you are done, "
            f"or dogma resolve {first.problem_id} --abandon to stop it."
        )
    else:
        tail = (
            f" Start with dogma resolve {first.problem_id}, then dogma submit "
            f"{first.problem_id} when you have a solution."
        )
    return "Timed re-solves, no hints: " + "; ".join(parts) + "." + tail


def _timed_action(item: ResolveDue) -> str:
    if item.open:
        return (
            f"Finish the open timed re-solve: dogma submit {item.problem_id}, "
            f"or dogma resolve {item.problem_id} --abandon."
        )
    return (
        f"Start the timed re-solve: dogma resolve {item.problem_id} "
        f"(budget {item.minutes} min, no hints)."
    )


def _review_or_resolve_action(due: int, familiar: FamiliarProblem | None, limit: int) -> str:
    if due:
        return _review_command(due, limit)
    if familiar is not None:
        return f"Open {familiar.problem_id} and read example 1."
    return "Nothing is due and nothing solved is ready. Finish with dogma close."


def _review_detail(due: int, limit: int) -> str:
    if due == 0:
        return "No cards are due right now."
    return (
        f"Answer up to {min(due, limit)} due cards ({due} due now): {_review_command(due, limit)}"
    )


def _review_command(due: int, limit: int) -> str:
    return f"dogma review --limit {min(due, limit)}"


def _next_topic(
    facts: PlanFacts, exclude: frozenset[str]
) -> tuple[tuple[TopicFacts, ProblemFacts] | None, str]:
    """The next topic to introduce, or None with the reason nothing qualifies."""
    eligible: list[tuple[int, int, TopicFacts, ProblemFacts]] = []
    for index, topic in enumerate(facts.topics):
        if topic.topic_id in exclude or _solved(facts, topic.topic_id):
            continue
        core = topic.ready_core()
        if core is None or any(not _solved(facts, req) for req in topic.requires):
            continue
        # Unfinished topics come before new ones; path order breaks ties.
        in_progress = facts.mastery.get(topic.topic_id) is Mastery.INTRODUCED
        eligible.append((0 if in_progress else 1, index, topic, core))
    if not eligible:
        return None, _why_nothing_is_new(facts, exclude)
    _, _, topic, core = min(eligible, key=lambda item: (item[0], item[1]))
    return (topic, core), ""


def _why_nothing_is_new(facts: PlanFacts, exclude: frozenset[str]) -> str:
    pending = [topic for topic in facts.topics if not _solved(facts, topic.topic_id)]
    if not pending:
        return "every topic on the core path is already solved"
    waiting = [topic for topic in pending if topic.topic_id not in exclude]
    if not waiting:
        return "the remaining topics are already planned for today"
    first = waiting[0]
    if first.ready_core() is None:
        return f"{first.topic_id} has no verified or reviewed content yet"
    missing = [req for req in first.requires if not _solved(facts, req)]
    if missing:
        return f"{first.topic_id} needs {missing[0]} solved first"
    return f"{first.topic_id} is already planned for today"


def _familiar_choice(facts: PlanFacts, exclude: frozenset[str]) -> FamiliarProblem | None:
    return next((item for item in facts.familiar if item.topic_id not in exclude), None)


def _solved(facts: PlanFacts, topic_id: str) -> bool:
    return mastery_rank(facts.mastery.get(topic_id, Mastery.UNSEEN)) >= mastery_rank(Mastery.SOLVED)
