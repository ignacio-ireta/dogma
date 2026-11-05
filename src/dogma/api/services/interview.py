"""Interview questions, rubrics and self-scored answers over HTTP (the ``dogma answer`` use cases).

The rules live in ``dogma.learning.answers`` and the rubric loader. This module maps them to the
wire shapes. The question's 60-second answer is never sent: the learner writes it, as in the CLI.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from dogma.api.ids import is_topic_id
from dogma.api.schemas.interview import (
    AnchorOut,
    AnswerOut,
    AnswerRequest,
    BandOut,
    DimensionOut,
    InterviewOut,
    InterviewQuestionOut,
    ItemOut,
    PartOut,
    RubricOut,
)
from dogma.content.interview import load_question, parse_questions
from dogma.content.rubrics import Rubric, load_rubric_version
from dogma.core.config import Settings
from dogma.core.errors import ContentError, DogmaError, ErrorCode
from dogma.db.models import Mastery
from dogma.db.progress import read_snapshot
from dogma.learning.answers import (
    TOPIC_ANSWER_ID,
    TOPIC_ANSWER_VERSION,
    AnswerResult,
    record_answer,
)
from dogma.learning.mocks import MOCK_RUBRIC_ID, MOCK_RUBRIC_VERSION
from dogma.learning.path import topic_row


def interview(settings: Settings, topic_id: str, *, now: datetime) -> InterviewOut:
    """The topic's numbered questions (titles only), its mastery and the topic-answer rubric."""
    directory = _topic_dir(settings, topic_id)
    path = directory / "interview.md"
    if not path.is_file():
        raise ContentError(ErrorCode.CONTENT_NOT_FOUND, f"{topic_id} has no interview questions")
    questions = parse_questions(path.read_text(encoding="utf-8"))
    snapshot = read_snapshot(settings, now=now)
    return InterviewOut(
        topic_id=topic_id,
        title=topic_row(settings, topic_id).title,
        mastery=snapshot.mastery.get(topic_id, Mastery.UNSEEN.value),
        questions=[
            InterviewQuestionOut(number=item.number, title=item.title) for item in questions
        ],
        rubric=rubric_out(_topic_answer_rubric(settings)),
    )


def answer(settings: Settings, topic_id: str, body: AnswerRequest, *, now: datetime) -> AnswerOut:
    """Store one self-scored answer and apply the interview-ready rule (``dogma answer``)."""
    directory = _topic_dir(settings, topic_id)
    if body.spoken and body.text is not None:
        raise DogmaError(
            ErrorCode.ANSWER_TEXT_INVALID, "give the answer as text or as spoken, not both."
        )
    if not body.spoken and body.text is None:
        raise DogmaError(
            ErrorCode.ANSWER_TEXT_INVALID,
            "write an answer of 1 to 8000 characters, or mark it as spoken.",
        )
    question = load_question(directory, body.question)
    result = record_answer(
        settings,
        topic_id=topic_id,
        prompt_ref=f"interview.md#Q{question.number}",
        text=None if body.spoken else body.text,
        scores=body.scores,
        notes=body.notes,
        now=now,
    )
    return answer_out(result)


def rubric(settings: Settings, rubric_id: str) -> RubricOut:
    """A rubric by id (``topic-answer`` or ``mock``), at the version the app scores with."""
    if rubric_id == TOPIC_ANSWER_ID:
        return rubric_out(_topic_answer_rubric(settings))
    if rubric_id == MOCK_RUBRIC_ID:
        return rubric_out(
            load_rubric_version(settings.curriculum_dir, MOCK_RUBRIC_ID, MOCK_RUBRIC_VERSION)
        )
    raise ContentError(ErrorCode.CONTENT_NOT_FOUND, f"no rubric {rubric_id!r}")


def answer_out(result: AnswerResult) -> AnswerOut:
    return AnswerOut(
        answer_id=result.answer_id,
        topic_id=result.topic_id,
        prompt_ref=result.prompt_ref,
        scores=dict(result.scores),
        spoken=result.spoken,
        before=result.before.value,
        after=result.after.value,
        promoted=result.promoted,
        ready_at=result.ready_at,
        reason=result.reason,
    )


def rubric_out(source: Rubric) -> RubricOut:
    """A rubric's anchors in level order, so the browser lists 1 to 4 without sorting."""
    return RubricOut(
        id=source.id,
        version=source.version,
        kind=source.kind,
        title=source.title,
        anchors=_anchors(source.scale.anchors),
        ready_score=source.ready_score,
        maximum=source.maximum,
        criteria=list(source.criteria_ids),
        dimensions=[
            DimensionOut(
                id=dimension.id,
                title=dimension.title,
                question=dimension.question,
                anchors=_anchors(dimension.anchors),
            )
            for dimension in source.dimensions
        ],
        parts=[
            PartOut(
                id=part.id,
                title=part.title,
                minutes=part.minutes,
                maximum=part.maximum,
                items=[ItemOut(id=item.id, text=item.text) for item in part.items],
            )
            for part in source.parts
        ],
        bands=[
            BandOut(label=band.label, low=band.low, high=band.high, text=band.text)
            for band in source.bands
        ],
    )


def _anchors(anchors: dict[int, str]) -> list[AnchorOut]:
    return [AnchorOut(level=level, text=anchors[level]) for level in sorted(anchors)]


def _topic_answer_rubric(settings: Settings) -> Rubric:
    return load_rubric_version(settings.curriculum_dir, TOPIC_ANSWER_ID, TOPIC_ANSWER_VERSION)


def _topic_dir(settings: Settings, topic_id: str) -> Path:
    if not is_topic_id(topic_id):
        raise ContentError(ErrorCode.CONTENT_NOT_FOUND, f"no topic {topic_id!r}")
    directory = settings.content_dir.joinpath(*topic_id.split("/"))
    if not (directory / "topic.yaml").is_file():
        raise ContentError(ErrorCode.CONTENT_NOT_FOUND, f"no topic {topic_id!r}")
    return directory
