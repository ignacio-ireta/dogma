"""Advisory model scores next to the learner's self-scores (I-18d, ADR-0007).

The self-score is the source of truth. A model score is stored in ``answer.model_score`` and is
only ever shown beside the self-score. Nothing in this module changes mastery, and the stored
payload is the one ``model_score_payload`` builds, so a reader can see which rubric and prompt
version produced it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from sqlalchemy import select, update

from dogma.core.config import Settings
from dogma.db.answer_models import Answer
from dogma.db.errors import StorageErrorCode
from dogma.db.session import open_session
from dogma.tutor.models import AdvisoryScores

NEAR_POINTS: Final = 1  # a difference of one point
FAR_POINTS: Final = 2  # a difference of two or more points


@dataclass(frozen=True)
class Comparison:
    """How the model's scores sit against the learner's, over the criteria both scored."""

    compared: int
    far: tuple[str, ...]  # two or more points apart
    near: tuple[str, ...]  # one point apart
    summary: str  # one line for the terminal


def compare(self_scores: Mapping[str, int], advisory: AdvisoryScores) -> Comparison:
    """Compare each criterion the model scored with the learner's own score for it."""
    pairs = [
        (item.criterion, abs(self_scores[item.criterion] - item.score))
        for item in advisory.scores
        if item.criterion in self_scores
    ]
    far = tuple(criterion for criterion, gap in pairs if gap >= FAR_POINTS)
    near = tuple(criterion for criterion, gap in pairs if gap == NEAR_POINTS)
    return Comparison(
        compared=len(pairs),
        far=far,
        near=near,
        summary=_summary(len(pairs), far, near),
    )


def model_score_payload(
    advisory: AdvisoryScores, *, rubric_id: str, rubric_version: int
) -> dict[str, object]:
    """The JSON stored on the answer: the model's scores and quotes, and where they came from."""
    return {
        "model": advisory.model,
        "prompt_version": advisory.prompt_version,
        "rubric_id": rubric_id,
        "rubric_version": rubric_version,
        "dropped": advisory.dropped,
        "scores": {
            item.criterion: {"score": item.score, "quote": item.quote} for item in advisory.scores
        },
    }


def attach_to_answer(settings: Settings, answer_id: str, payload: dict[str, object]) -> None:
    """Store the advisory payload on one answer. No other column changes."""
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.execute(update(Answer).where(Answer.id == answer_id).values(model_score=payload))


def attach_to_mock(settings: Settings, session_id: str, payload: dict[str, object]) -> None:
    """Store the advisory payload on the scored answer of one mock."""
    with open_session(settings, on_failure=StorageErrorCode.DB_WRITE_FAILED) as session:
        session.execute(
            update(Answer).where(Answer.mock_session_id == session_id).values(model_score=payload)
        )


def mock_answer_text(settings: Settings, session_id: str) -> str | None:
    """The item notes stored for a scored mock: the only learner text a mock has."""
    with open_session(settings, on_failure=StorageErrorCode.DB_READ_FAILED) as session:
        return session.scalars(
            select(Answer.text).where(Answer.mock_session_id == session_id)
        ).first()


def _summary(compared: int, far: tuple[str, ...], near: tuple[str, ...]) -> str:
    if compared == 0:
        return (
            "The model gave no scores with a quote from your text, so there is nothing to "
            "compare. Your scores stand."
        )
    if not far and not near:
        return f"The model agrees with your scores on all {compared} it scored. Your scores stand."
    parts: list[str] = []
    if far:
        parts.append(f"by 2 or more on {', '.join(far)}")
    if near:
        parts.append(f"by 1 on {', '.join(near)}")
    return f"The model differs from you {', and '.join(parts)}. Your scores stand."
