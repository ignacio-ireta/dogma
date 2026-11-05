"""Interview questions in a topic's interview.md: numbered H2 sections, read by number."""

from __future__ import annotations

from pathlib import Path

import pytest

from dogma.content.interview import load_question, parse_questions
from dogma.core.errors import ContentError, ErrorCode

REPO = Path(__file__).resolve().parents[3]
EXAMPLE_INTERVIEW = REPO / "content" / "cml" / "vector-operations" / "interview.md"


def test_the_example_topic_has_three_numbered_questions() -> None:
    questions = parse_questions(EXAMPLE_INTERVIEW.read_text(encoding="utf-8"))

    assert [question.number for question in questions] == [1, 2, 3]
    assert questions[0].title == "What does the dot product measure, geometrically?"
    assert "60-second answer" in questions[0].body


def test_headings_that_are_not_numbered_questions_are_skipped() -> None:
    text = "# Notes\n\n## Common mistakes\n\n- one\n\n## Q2. Second?\n\nBody.\n"

    questions = parse_questions(text)

    assert [(q.number, q.title) for q in questions] == [(2, "Second?")]


def test_a_fenced_block_with_a_hash_heading_is_not_a_question() -> None:
    text = "## Q1. Real?\n\n```\n## Q9. Not a heading\n```\n"

    assert [q.number for q in parse_questions(text)] == [1]


def test_load_question_returns_the_named_question(tmp_path: Path) -> None:
    (tmp_path / "interview.md").write_text(
        "## Q1. One?\n\nA.\n\n## Q2. Two?\n\nB.\n", encoding="utf-8"
    )

    question = load_question(tmp_path, 2)

    assert question.title == "Two?"


@pytest.mark.parametrize("number", [0, 3])
def test_a_question_number_that_does_not_exist_is_a_content_error(
    tmp_path: Path, number: int
) -> None:
    (tmp_path / "interview.md").write_text("## Q1. One?\n\nA.\n", encoding="utf-8")

    with pytest.raises(ContentError) as caught:
        load_question(tmp_path, number)

    assert caught.value.code is ErrorCode.CONTENT_NOT_FOUND


def test_a_topic_without_an_interview_file_is_a_content_error(tmp_path: Path) -> None:
    with pytest.raises(ContentError) as caught:
        load_question(tmp_path, 1)

    assert caught.value.code is ErrorCode.CONTENT_NOT_FOUND
