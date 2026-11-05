"""Case packages (CONTENT_SPEC §9): one failing fixture test per case rule.

The case fixture lives under tests/fixtures/content/cases/tst/casey and is checked against its own
syllabus (tests/fixtures/content/case-syllabus.yaml), so the topic fixtures stay unchanged.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from dogma.content.loader import load_syllabus, load_topic
from dogma.content.schemas import Syllabus
from dogma.content.validate import Issue, validate_all, validate_topic
from dogma.core.errors import ErrorCode

Edit = Callable[[Path, str, str], None]
WriteSyllabus = Callable[[str], Syllabus]


@pytest.fixture
def case_syllabus(fixtures: Path) -> Syllabus:
    return load_syllabus(fixtures / "case-syllabus.yaml")


@pytest.fixture
def casey(tmp_path: Path, fixtures: Path) -> Path:
    """A writable copy of the case fixture, at a path that ends with tst/casey."""
    target = tmp_path / "cases" / "tst" / "casey"
    shutil.copytree(fixtures / "cases" / "tst" / "casey", target)
    return target


def _codes(issues: list[Issue]) -> set[str]:
    return {issue.code for issue in issues}


def _case_codes(topic_dir: Path, syllabus: Syllabus) -> set[str]:
    return _codes(validate_topic(load_topic(topic_dir), syllabus))


def _write_reveals(casey: Path, body: str) -> None:
    (casey / "reveals.yaml").write_text(body, encoding="utf-8")


def test_fixture_case_package_has_no_issues(casey: Path, case_syllabus: Syllabus) -> None:
    assert validate_topic(load_topic(casey), case_syllabus) == []


def test_loader_reads_kind_case_and_its_parts(casey: Path) -> None:
    package = load_topic(casey)
    assert package.meta.kind == "case"
    assert package.case is not None
    assert package.case.reveals is not None and len(package.case.reveals) == 2
    assert package.case.exhibits == ("exhibits/counts.csv",)
    assert package.problems == ()


# --- required files ------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["brief.md", "reveals.yaml", "twist.md", "debrief.md"])
def test_case_file_is_required(casey: Path, case_syllabus: Syllabus, name: str) -> None:
    (casey / name).unlink()
    assert "CASE_FILE_MISSING" in _case_codes(casey, case_syllabus)


def test_case_lesson_follows_the_same_sections_as_a_topic(
    casey: Path, case_syllabus: Syllabus, edit: Edit
) -> None:
    edit(casey / "lesson.md", "## Connections", "## Links")
    assert "LESSON_SECTIONS" in _case_codes(casey, case_syllabus)


def test_case_has_no_problems_directory(casey: Path, case_syllabus: Syllabus) -> None:
    (casey / "problems" / "p01-todo").mkdir(parents=True)
    assert "CASE_HAS_PROBLEMS" in _case_codes(casey, case_syllabus)


def test_interview_is_optional_for_a_case(casey: Path, case_syllabus: Syllabus) -> None:
    assert not (casey / "interview.md").exists()
    assert "INTERVIEW_MISSING" not in _case_codes(casey, case_syllabus)


# --- debrief.md ----------------------------------------------------------------------------


def test_debrief_needs_the_four_rass_parts_in_order(
    casey: Path, case_syllabus: Syllabus, edit: Edit
) -> None:
    edit(casey / "debrief.md", "## Ask", "## Questions")
    assert "CASE_DEBRIEF_SECTIONS" in _case_codes(casey, case_syllabus)


def test_debrief_sections_must_not_be_empty(casey: Path, case_syllabus: Syllabus) -> None:
    (casey / "debrief.md").write_text(
        "# Casey debrief\n\n## Reframe\n\nRestate the decision.\n\n## Ask\n\n"
        "## Structure\n\nRank by risk.\n\n## Synthesize\n\nPilot it.\n",
        encoding="utf-8",
    )
    assert "CASE_TEXT_EMPTY" in _case_codes(casey, case_syllabus)


# --- brief.md and twist.md -----------------------------------------------------------------


def test_brief_must_not_be_empty(casey: Path, case_syllabus: Syllabus) -> None:
    (casey / "brief.md").write_text("  \n", encoding="utf-8")
    assert "CASE_TEXT_EMPTY" in _case_codes(casey, case_syllabus)


def test_twist_must_not_be_empty(casey: Path, case_syllabus: Syllabus) -> None:
    (casey / "twist.md").write_text("", encoding="utf-8")
    assert "CASE_TEXT_EMPTY" in _case_codes(casey, case_syllabus)


def test_todo_placeholders_block_verification(
    casey: Path, case_syllabus: Syllabus, edit: Edit
) -> None:
    edit(casey / "brief.md", "What would you build", "TODO: what would you build")
    assert "CASE_TODO" in _case_codes(casey, case_syllabus)


# --- reveals.yaml --------------------------------------------------------------------------


def test_reveals_need_two_to_eight_entries(casey: Path, case_syllabus: Syllabus) -> None:
    _write_reveals(
        casey,
        "- when_asked: How many churned?\n  answer: Thirty.\n",
    )
    assert "CASE_REVEAL_COUNT" in _case_codes(casey, case_syllabus)


def test_reveal_exhibit_must_exist(casey: Path, case_syllabus: Syllabus) -> None:
    (casey / "exhibits" / "counts.csv").unlink()
    assert "CASE_EXHIBIT_MISSING" in _case_codes(casey, case_syllabus)


def test_reveal_exhibit_path_must_be_under_exhibits(casey: Path, case_syllabus: Syllabus) -> None:
    text = (casey / "reveals.yaml").read_text(encoding="utf-8")
    _write_reveals(casey, text.replace("exhibits/counts.csv", "../counts.csv"))
    issues = validate_all(casey.parent.parent, case_syllabus)
    assert [issue.code for issue in issues] == [ErrorCode.CONTENT_SCHEMA_INVALID.value]


def test_reveal_fields_are_strict(casey: Path, case_syllabus: Syllabus) -> None:
    _write_reveals(
        casey,
        "- when_asked: How many churned?\n  answer: Thirty.\n  hint: ask twice\n"
        "- when_asked: Which data?\n  answer: Tenure.\n",
    )
    issues = validate_all(casey.parent.parent, case_syllabus)
    assert [issue.code for issue in issues] == [ErrorCode.CONTENT_SCHEMA_INVALID.value]


def test_reveal_answer_must_not_be_empty(casey: Path, case_syllabus: Syllabus) -> None:
    _write_reveals(
        casey,
        "- when_asked: How many churned?\n  answer: ''\n"
        "- when_asked: Which data?\n  answer: Tenure.\n",
    )
    issues = validate_all(casey.parent.parent, case_syllabus)
    assert [issue.code for issue in issues] == [ErrorCode.CONTENT_SCHEMA_INVALID.value]


# --- exhibits/ -----------------------------------------------------------------------------


def test_exhibits_are_png_or_csv_only(casey: Path, case_syllabus: Syllabus) -> None:
    (casey / "exhibits" / "notes.txt").write_text("not an exhibit", encoding="utf-8")
    assert "CASE_EXHIBIT_TYPE" in _case_codes(casey, case_syllabus)


def test_exhibits_are_flat_files_not_folders(casey: Path, case_syllabus: Syllabus) -> None:
    (casey / "exhibits" / "extra").mkdir()
    assert "CASE_EXHIBIT_TYPE" in _case_codes(casey, case_syllabus)


# --- topic.yaml kind -----------------------------------------------------------------------


def test_kind_must_match_the_syllabus_entry(casey: Path, write_syllabus: WriteSyllabus) -> None:
    syllabus = write_syllabus(
        "version: 1\nsource: test\ntracks:\n  - code: tst\n    order: 1\n    title: T\n"
        "    origin: local\n    topic_count: 1\n    topics:\n"
        "      - {id: tst/casey, title: 'Casey case', tier: B}\n"
    )
    assert "TOPIC_KIND_MISMATCH" in _case_codes(casey, syllabus)


def test_a_topic_cannot_carry_case_files(alpha: Path, syllabus: Syllabus) -> None:
    (alpha / "brief.md").write_text("# Stray brief\n", encoding="utf-8")
    assert "TOPIC_CASE_FILES" in _codes(validate_topic(load_topic(alpha), syllabus))


def test_a_topic_still_needs_its_problems(alpha: Path, syllabus: Syllabus) -> None:
    # A topic keeps the 1-3 problem rule; kind: case is the only way to have none.
    shutil.rmtree(alpha / "problems")
    assert "TOPIC_PROBLEM_COUNT" in _codes(validate_topic(load_topic(alpha), syllabus))
