"""Interviewer mode: the model picks one reveal for a question, or none (ARCHITECTURE §4.5).

The model sees the case brief and the questions the reveals answer, numbered. It never sees
the reveal texts or debrief.md. It replies with one JSON object holding a number or null, and
the caller prints the reveal's own text from reveals.yaml. The model's words are never printed,
so it cannot release a fact that the file does not hold. Whether the question earns the chosen
reveal is checked by the caller (learning.mocks), not by the model.
"""

from __future__ import annotations

from collections.abc import Sequence

from dogma.core.errors import DogmaError, ErrorCode
from dogma.tutor.gateway import CallMeta, Gateway
from dogma.tutor.models import CallMode, PickedReveal
from dogma.tutor.prompts import render_prompt
from dogma.tutor.replies import json_object

INTERVIEWER_MAX_TOKENS = 60
MAX_QUESTION_CHARS = 500


def pick_reveal(
    gateway: Gateway, *, brief: str, when_asked: Sequence[str], question: str
) -> PickedReveal:
    """Ask which numbered reveal answers ``question``. Raises for an invalid question or reply."""
    clean = clean_question(question)
    prompt = render_prompt(CallMode.INTERVIEWER, None, build_user_text(brief, when_asked, clean))
    reply = gateway.complete(
        CallMeta(mode=CallMode.INTERVIEWER, prompt=prompt, max_tokens=INTERVIEWER_MAX_TOKENS)
    )
    number = _number(reply.text, count=len(when_asked))
    return PickedReveal(number=number, model=reply.model, prompt_version=prompt.version)


def clean_question(question: str) -> str:
    """The learner's question, trimmed. Empty or over-long questions never reach a model."""
    stripped = question.strip()
    if not stripped or len(stripped) > MAX_QUESTION_CHARS:
        raise DogmaError(
            ErrorCode.TUTOR_QUESTION_INVALID,
            f"ask a question of 1 to {MAX_QUESTION_CHARS} characters.",
        )
    return stripped


def build_user_text(brief: str, when_asked: Sequence[str], question: str) -> str:
    """The case brief, the numbered questions the reveals answer, and the learner's question."""
    numbered = "\n".join(f"{number}: {text}" for number, text in enumerate(when_asked, start=1))
    return "\n".join(
        [
            "TASK: pick the one numbered reveal that answers the CANDIDATE QUESTION, or null.",
            "",
            "### CASE BRIEF",
            brief.strip(),
            "",
            "### REVEALS (number: the question each one answers)",
            numbered or "(none)",
            "",
            "### CANDIDATE QUESTION",
            question,
        ]
    )


def _number(reply: str, *, count: int) -> int | None:
    data = json_object(
        reply,
        ErrorCode.TUTOR_INTERVIEWER_REPLY_INVALID,
        "the interviewer's reply was not a numbered reveal or null",
    )
    if set(data) != {"reveal"}:
        raise _invalid()
    value = data["reveal"]
    if value is None:
        return None
    if type(value) is not int or not 1 <= value <= count:
        raise _invalid()
    return value


def _invalid() -> DogmaError:
    return DogmaError(
        ErrorCode.TUTOR_INTERVIEWER_REPLY_INVALID,
        "the interviewer's reply was not a numbered reveal or null",
    )
