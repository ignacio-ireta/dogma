"""Numbered interview questions in a topic's ``interview.md`` (ARCHITECTURE §4.1).

Each question is an H2 heading ``Q<n>. <question>``. Other H2 sections (such as "Common mistakes")
are not questions and are skipped. The 60-second answer is the learner's to produce, so the
parser keeps the full body only for the tests and the author tools; the CLI shows the title.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from dogma.content.markdown import parse_outline
from dogma.core.errors import ContentError, ErrorCode

INTERVIEW_FILENAME = "interview.md"
_QUESTION = re.compile(r"^Q(\d+)\.\s+(.+)$")


@dataclass(frozen=True)
class InterviewQuestion:
    """One numbered question. ``body`` is the section text under its heading."""

    number: int
    title: str
    body: str


def parse_questions(text: str) -> tuple[InterviewQuestion, ...]:
    """Every ``Q<n>.`` section in document order. Other headings are ignored."""
    questions: list[InterviewQuestion] = []
    for section in parse_outline(text).sections:
        match = _QUESTION.match(section.heading)
        if match is not None:
            questions.append(
                InterviewQuestion(
                    number=int(match.group(1)), title=match.group(2).strip(), body=section.text
                )
            )
    return tuple(questions)


def load_question(topic_dir: Path, number: int) -> InterviewQuestion:
    """Question ``number`` of the topic in ``topic_dir``. A missing one is a content error."""
    path = topic_dir / INTERVIEW_FILENAME
    if not path.is_file():
        raise ContentError(ErrorCode.CONTENT_NOT_FOUND, f"{path}: no interview questions")
    for question in parse_questions(path.read_text(encoding="utf-8")):
        if question.number == number:
            return question
    raise ContentError(ErrorCode.CONTENT_NOT_FOUND, f"{path} has no question Q{number}")
