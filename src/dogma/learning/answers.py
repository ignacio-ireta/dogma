"""Interview answers and the interview-ready rule (ARCHITECTURE §4.4, ADR-0007).

The learner answers a topic's interview question and scores it on the ``topic-answer`` rubric.
The score is self-given and it is the source of truth. A retained topic becomes interview-ready
when one such answer scores at least the rubric's ready score on every dimension. Answers never
lower mastery: a failed timed re-solve or a card lapse does that, through the rules in
``dogma.learning.resolves`` and ``dogma.learning.review``.

Only a ``retained`` topic can be promoted. An answer given earlier, while the topic was still
consolidating, does not count later: the learner answers again once the topic is retained. That
keeps the evidence tied to the state it promotes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Final
from uuid import uuid4

from dogma.content.rubrics import Rubric, check_scores, load_rubric_version
from dogma.core.config import Settings
from dogma.core.errors import ContentError, DogmaError, ErrorCode
from dogma.db.answer_models import Answer, AnswerKind
from dogma.db.errors import StorageErrorCode
from dogma.db.models import Mastery, TopicState
from dogma.db.session import open_session
from dogma.learning.path import topic_row

TOPIC_ANSWER_ID: Final = "topic-answer"
TOPIC_ANSWER_VERSION: Final = 1
MAX_TEXT: Final = 8000
MAX_NOTES: Final = 1000


@dataclass(frozen=True)
class AnswerResult:
    """What recording an answer stored, and what it did to the topic's mastery."""

    answer_id: str
    topic_id: str
    prompt_ref: str
    scores: Mapping[str, int]
    spoken: bool
    before: Mastery
    after: Mastery
    ready_at: datetime | None
    reason: str

    @property
    def promoted(self) -> bool:
        return self.after is not self.before


def is_ready_answer(scores: Mapping[str, int], ready_score: int) -> bool:
    """True when every dimension reaches the ready score."""
    return bool(scores) and min(scores.values()) >= ready_score


def readiness_change(current: Mastery, scores: Mapping[str, int], ready_score: int) -> Mastery:
    """The mastery after one answer. Only a retained topic with a qualifying answer moves."""
    if current is Mastery.RETAINED and is_ready_answer(scores, ready_score):
        return Mastery.INTERVIEW_READY
    return current


def readiness_reason(
    before: Mastery, after: Mastery, scores: Mapping[str, int], ready_score: int
) -> str:
    """One line: what the answer did to the topic, or why it did nothing."""
    if after is Mastery.INTERVIEW_READY and before is not after:
        return f"Interview-ready: retained, and every dimension scored {ready_score} or higher."
    if before is Mastery.INTERVIEW_READY:
        return (
            "Still interview-ready. An answer never lowers mastery; a failed timed re-solve or "
            "a card lapse does."
        )
    if before is Mastery.RETAINED:
        below = ", ".join(sorted(key for key, value in scores.items() if value < ready_score))
        return (
            f"Still retained: {below} scored below {ready_score}. Interview-ready needs "
            f"{ready_score} or higher on every dimension."
        )
    if before is Mastery.UNSEEN:
        return "Not started: interview-ready needs a retained topic."
    return (
        f"Still {before.value}: interview-ready needs a retained topic. Answer again once it "
        "is retained."
    )


def record_answer(
    settings: Settings,
    *,
    topic_id: str,
    prompt_ref: str,
    text: str | None,
    scores: Mapping[str, int],
    now: datetime,
    notes: str | None = None,
) -> AnswerResult:
    """Store one self-scored answer and apply the interview-ready rule, in one transaction.

    ``text`` None means the answer was given aloud. Refused while recording is off, so nothing is
    written then. The scores must match the rubric exactly: every dimension, 1 to 4, nothing else.
    """
    if not settings.record_attempts:
        raise DogmaError(
            ErrorCode.ANSWER_RECORDING_OFF,
            "Recording is switched off (DOGMA_RECORD_ATTEMPTS=false), so the answer was not "
            "stored. Turn recording on to record an answer.",
        )
    if topic_row(settings, topic_id).status == "missing":
        raise ContentError(ErrorCode.CONTENT_NOT_FOUND, f"no topic {topic_id}")
    rubric = load_rubric_version(settings.curriculum_dir, TOPIC_ANSWER_ID, TOPIC_ANSWER_VERSION)
    problems = check_scores(rubric, scores)
    if problems:
        raise DogmaError(ErrorCode.ANSWER_SCORE_INVALID, "; ".join(problems) + ".")
    clean_text = _clean_text(text)
    clean_notes = _clean_notes(notes)
    stored_scores = {key: int(scores[key]) for key in rubric.criteria_ids}
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        state = session.get(TopicState, topic_id)
        before = Mastery.UNSEEN if state is None else Mastery(state.mastery)
        after = readiness_change(before, stored_scores, _ready_score(rubric))
        answer = Answer(
            id=uuid4().hex,
            topic_id=topic_id,
            kind=AnswerKind.INTERVIEW.value,
            prompt_ref=prompt_ref,
            text=clean_text,
            rubric_id=rubric.id,
            rubric_version=rubric.version,
            scores=stored_scores,
            notes=clean_notes,
            created_at=now,
        )
        session.add(answer)
        ready_at = None
        if state is not None and after is not before:
            state.mastery = after.value
            state.ready_at = now
            ready_at = now
        answer_id = answer.id
    return AnswerResult(
        answer_id=answer_id,
        topic_id=topic_id,
        prompt_ref=prompt_ref,
        scores=stored_scores,
        spoken=clean_text is None,
        before=before,
        after=after,
        ready_at=ready_at,
        reason=readiness_reason(before, after, stored_scores, _ready_score(rubric)),
    )


def _ready_score(rubric: Rubric) -> int:
    if rubric.ready_score is None:
        raise ContentError(ErrorCode.CONTENT_SCHEMA_INVALID, f"{rubric.id} has no ready score")
    return rubric.ready_score


def _clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped or len(stripped) > MAX_TEXT:
        raise DogmaError(
            ErrorCode.ANSWER_TEXT_INVALID,
            f"write an answer of 1 to {MAX_TEXT} characters, or give it aloud with --spoken.",
        )
    return stripped


def _clean_notes(notes: str | None) -> str | None:
    if notes is None or not notes.strip():
        return None
    if len(notes) > MAX_NOTES:
        raise DogmaError(
            ErrorCode.ANSWER_TEXT_INVALID, f"notes are at most {MAX_NOTES} characters."
        )
    return notes.strip()
