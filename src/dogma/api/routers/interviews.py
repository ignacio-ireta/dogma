"""Interview questions, the two rubrics and self-scored answers (ARCHITECTURE §4.4, §4.7)."""

from __future__ import annotations

from fastapi import APIRouter

from dogma.api.deps import ClockDep, SettingsDep
from dogma.api.openapi import error_responses
from dogma.api.schemas.interview import AnswerOut, AnswerRequest, InterviewOut, RubricOut
from dogma.api.services import interview

# Registered before the content router: GET /topics/{id}/interview must not be read as a topic id.
router = APIRouter(tags=["interview"])


@router.get(
    "/topics/{topic_id:path}/interview",
    response_model=InterviewOut,
    summary="A topic's interview questions (titles), its mastery and the topic-answer rubric",
    description=(
        "The 60-second answers are not sent: you write them, as `dogma answer` shows only the "
        "question. The rubric lists each dimension's anchors so you score against the words."
    ),
    responses=error_responses(
        (404, "No topic with this id, or it has no interview questions"),
    ),
)
def get_interview(topic_id: str, settings: SettingsDep, clock: ClockDep) -> InterviewOut:
    return interview.interview(settings, topic_id, now=clock())


@router.post(
    "/topics/{topic_id:path}/answers",
    response_model=AnswerOut,
    status_code=201,
    summary="Record a self-scored answer to a topic's question (`dogma answer`)",
    description=(
        "Give `text` (written) or `spoken: true` (said aloud, no text kept). Scores are 1 to 4 on "
        "every dimension. The response says what the answer did to the topic's mastery. Only a "
        "retained topic can become interview-ready."
    ),
    responses=error_responses(
        (404, "No topic, or no such question number"),
        (409, "Recording is switched off"),
        (422, "Scores incomplete or out of range, or the answer text is missing or too long"),
    ),
)
def post_answer(
    topic_id: str, body: AnswerRequest, settings: SettingsDep, clock: ClockDep
) -> AnswerOut:
    return interview.answer(settings, topic_id, body, now=clock())


@router.get(
    "/rubrics/{rubric_id}",
    response_model=RubricOut,
    summary="A scoring rubric: `topic-answer` (per topic) or `mock` (the 30-item mock rubric)",
    responses=error_responses((404, "No rubric with this id")),
)
def get_rubric(rubric_id: str, settings: SettingsDep) -> RubricOut:
    return interview.rubric(settings, rubric_id)
