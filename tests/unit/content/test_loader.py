from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from dogma.content.loader import (
    discover_topic_dirs,
    load_core_path,
    load_problem,
    load_syllabus,
    load_topic,
)
from dogma.content.schemas import CorePath, Syllabus
from dogma.core.errors import ContentError, ErrorCode


def test_real_core_path_loads_164_unique_topics(real_core: CorePath) -> None:
    listed = [topic for week in real_core.weeks for topic in week.topics]
    assert len(listed) == len(set(listed)) == 164


def test_real_example_package_loads_all_parts(real_example: Path) -> None:
    pkg = load_topic(real_example)
    assert pkg.meta.id == "cml/vector-operations"
    assert pkg.lesson is not None
    assert pkg.lesson_front_matter is not None and pkg.lesson_front_matter.minutes == 12
    assert pkg.cards is not None and len(pkg.cards) == 6
    assert pkg.interview_present
    assert [problem.dir.name for problem in pkg.problems] == ["p01-vector-projection"]
    assert pkg.problems[0].meta is not None and pkg.problems[0].meta.core


def test_fixture_topic_reads_lesson_sections_in_order(alpha: Path) -> None:
    pkg = load_topic(alpha)
    assert pkg.lesson is not None
    assert pkg.lesson.headings == (
        "The system",
        "Worked example",
        "Why it works",
        "Signal vs. noise",
        "Where it breaks",
        "Interview angle",
        "Connections",
    )
    assert pkg.interview is not None
    assert pkg.interview.headings == (
        "Q1. What does the doubling function do?",
        "Q2. Why is mutation a problem here?",
        "Common mistakes",
    )


def test_fixture_problem_records_files_hints_and_python_facts(alpha_problem: Path) -> None:
    problem = load_problem(alpha_problem)
    assert problem.meta is not None and problem.meta.entry == "double_all"
    assert "solution.py" in problem.files
    assert problem.hints is not None
    assert problem.hints.headings[0] == "Hint 1: concept"
    assert problem.python["solution.py"].defined == frozenset({"double_all"})
    assert len(problem.python["test_hidden.py"].test_functions) == 5


def test_optional_topic_files_may_be_absent(alpha: Path) -> None:
    (alpha / "lesson.md").unlink()
    (alpha / "cards.yaml").unlink()
    (alpha / "interview.md").unlink()
    pkg = load_topic(alpha)
    assert pkg.lesson is None and pkg.lesson_front_matter is None
    assert pkg.cards is None
    assert pkg.interview is None and not pkg.interview_present


def test_problem_without_meta_yaml_loads_with_meta_none(alpha_problem: Path) -> None:
    (alpha_problem / "meta.yaml").unlink()
    assert load_problem(alpha_problem).meta is None


def test_missing_file_raises_content_not_found(tmp_path: Path) -> None:
    with pytest.raises(ContentError) as excinfo:
        load_syllabus(tmp_path / "nope.yaml")
    assert excinfo.value.code == ErrorCode.CONTENT_NOT_FOUND


def test_invalid_yaml_raises_schema_invalid_with_the_path(tmp_path: Path) -> None:
    bad = tmp_path / "syllabus.yaml"
    bad.write_text("tracks: [unclosed\n", encoding="utf-8")
    with pytest.raises(ContentError) as excinfo:
        load_syllabus(bad)
    assert excinfo.value.code == ErrorCode.CONTENT_SCHEMA_INVALID
    assert str(bad) in excinfo.value.message
    assert "invalid YAML" in excinfo.value.message


def test_schema_error_names_the_field(alpha: Path, edit: Callable[[Path, str, str], None]) -> None:
    edit(alpha / "topic.yaml", "tier: A", "tier: Z")
    with pytest.raises(ContentError) as excinfo:
        load_topic(alpha)
    assert excinfo.value.code == ErrorCode.CONTENT_SCHEMA_INVALID
    assert "tier" in excinfo.value.message
    assert str(alpha / "topic.yaml") in excinfo.value.message


def test_lesson_without_front_matter_is_a_schema_error(alpha: Path) -> None:
    (alpha / "lesson.md").write_text("## The system\n\nNo front matter here.\n", encoding="utf-8")
    with pytest.raises(ContentError) as excinfo:
        load_topic(alpha)
    assert excinfo.value.code == ErrorCode.CONTENT_SCHEMA_INVALID
    assert "front matter" in excinfo.value.message


def test_discover_finds_exactly_the_topic_directories(packages: Path) -> None:
    assert discover_topic_dirs(packages) == [packages / "tst" / "alpha", packages / "tst" / "beta"]


def test_discover_on_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ContentError) as excinfo:
        discover_topic_dirs(tmp_path / "missing")
    assert excinfo.value.code == ErrorCode.CONTENT_NOT_FOUND


def test_load_core_path_reads_the_fixture(fixtures: Path) -> None:
    core = load_core_path(fixtures / "core.yaml")
    assert [topic for week in core.weeks for topic in week.topics] == ["tst/alpha", "tst/gamma"]


def test_real_syllabus_has_164_tier_a_topics(real_syllabus: Syllabus) -> None:
    tier_a = [topic for topic in real_syllabus.topics_by_id().values() if topic.tier == "A"]
    assert len(tier_a) == 164
