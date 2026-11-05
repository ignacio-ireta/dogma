"""One failing test per validation rule, each on a mutated copy of the fixture package."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from dogma.content.loader import load_topic
from dogma.content.schemas import CorePath, Syllabus
from dogma.content.validate import (
    Issue,
    validate_all,
    validate_path,
    validate_prereq_graph,
    validate_topic,
)
from dogma.core.errors import ErrorCode

Edit = Callable[[Path, str, str], None]

SYLLABUS_WITHOUT_ALPHA = """
version: 1
source: test
tracks:
  - code: tst
    order: 1
    title: Test track
    origin: deepml
    topic_count: 2
    topics:
      - {id: tst/beta, title: "Beta topic", tier: B}
      - {id: tst/gamma, title: "Gamma topic", tier: A}
"""

CORE_HEAD = """
id: core
title: Test core
start: 2026-10-26
end: 2026-11-01
sessions_per_day: 3
session_minutes: 45
days_per_week: 6
weeks:
  - week: 1
    starts: 2026-10-26
    theme: Test
    topics: ["""


def _codes(issues: list[Issue]) -> set[str]:
    return {issue.code for issue in issues}


def _topic_codes(topic_dir: Path, syllabus: Syllabus) -> set[str]:
    return _codes(validate_topic(load_topic(topic_dir), syllabus))


def _path_codes(topics: str, write_core: Callable[[str], CorePath], syllabus: Syllabus) -> set[str]:
    return _codes(validate_path(write_core(CORE_HEAD + topics + "]\n"), syllabus))


def _cards(numbers: list[int]) -> str:
    body = "".join(
        f"  - id: tst/alpha#{number}\n    type: qa\n    front: Q{number}?\n    back: A{number}\n"
        for number in numbers
    )
    return "cards:\n" + body


# --- clean inputs --------------------------------------------------------------------------


def test_fixture_packages_have_no_issues(packages: Path, syllabus: Syllabus) -> None:
    assert validate_all(packages, syllabus) == []


def test_fixture_core_path_has_no_issues(core: CorePath, syllabus: Syllabus) -> None:
    assert validate_path(core, syllabus) == []


def test_real_example_package_has_no_issues(real_example: Path, real_syllabus: Syllabus) -> None:
    assert validate_topic(load_topic(real_example), real_syllabus) == []


def test_real_syllabus_and_core_path_have_no_issues(
    real_core: CorePath, real_syllabus: Syllabus
) -> None:
    assert validate_path(real_core, real_syllabus) == []


# --- topic.yaml against the syllabus -------------------------------------------------------


def test_topic_not_in_syllabus_is_reported(
    alpha: Path, write_syllabus: Callable[[str], Syllabus]
) -> None:
    syllabus = write_syllabus(SYLLABUS_WITHOUT_ALPHA)
    assert "TOPIC_NOT_IN_SYLLABUS" in _topic_codes(alpha, syllabus)


def test_title_must_match_the_syllabus(alpha: Path, syllabus: Syllabus, edit: Edit) -> None:
    edit(alpha / "topic.yaml", "title: Alpha topic", "title: Alpha")
    assert "TOPIC_TITLE_MISMATCH" in _topic_codes(alpha, syllabus)


def test_tier_must_match_the_syllabus(alpha: Path, syllabus: Syllabus, edit: Edit) -> None:
    edit(alpha / "topic.yaml", "tier: A", "tier: C")
    assert "TOPIC_TIER_MISMATCH" in _topic_codes(alpha, syllabus)


def test_topic_directory_must_end_with_its_id(alpha: Path, syllabus: Syllabus) -> None:
    moved = shutil.copytree(alpha, alpha.parent / "alpha-copy")
    assert "CONTENT_ID_MISMATCH" in _topic_codes(moved, syllabus)


def test_requires_must_name_syllabus_topics(beta: Path, syllabus: Syllabus, edit: Edit) -> None:
    edit(beta / "topic.yaml", "requires: [tst/alpha]", "requires: [tst/nowhere]")
    assert "TOPIC_REQUIRES_UNKNOWN" in _topic_codes(beta, syllabus)


def test_prerequisite_cycle_is_reported_with_its_path(
    packages: Path, syllabus: Syllabus, alpha: Path, edit: Edit
) -> None:
    edit(alpha / "topic.yaml", "requires: []", "requires: [tst/beta]")
    issues = validate_all(packages, syllabus)
    cycles = [issue for issue in issues if issue.code == ErrorCode.CONTENT_PREREQ_CYCLE.value]
    assert len(cycles) == 1
    assert "tst/alpha -> tst/beta -> tst/alpha" in cycles[0].message


def test_duplicate_topic_ids_across_packages_are_reported(alpha: Path) -> None:
    pkg = load_topic(alpha)
    assert "TOPIC_DUPLICATE" in _codes(validate_prereq_graph([pkg, pkg]))


def test_unreadable_topic_becomes_an_issue_not_an_exception(
    packages: Path, syllabus: Syllabus, alpha: Path
) -> None:
    (alpha / "topic.yaml").write_text("tier: [\n", encoding="utf-8")
    issues = validate_all(packages, syllabus)
    assert [issue.code for issue in issues] == ["CONTENT_SCHEMA_INVALID"]
    assert str(alpha) in issues[0].path


# --- lesson.md -----------------------------------------------------------------------------


def test_lesson_file_must_exist(alpha: Path, syllabus: Syllabus) -> None:
    (alpha / "lesson.md").unlink()
    assert "LESSON_MISSING" in _topic_codes(alpha, syllabus)


def test_lesson_sections_must_be_exact_and_in_order(
    alpha: Path, syllabus: Syllabus, edit: Edit
) -> None:
    edit(alpha / "lesson.md", "## Connections", "## Links")
    assert "LESSON_SECTIONS" in _topic_codes(alpha, syllabus)


# --- cards.yaml ----------------------------------------------------------------------------


def test_cards_file_must_exist(alpha: Path, syllabus: Syllabus) -> None:
    (alpha / "cards.yaml").unlink()
    assert "CARDS_MISSING" in _topic_codes(alpha, syllabus)


def test_card_count_must_be_four_to_eight(alpha: Path, syllabus: Syllabus) -> None:
    for count in (3, 9):
        (alpha / "cards.yaml").write_text(_cards(list(range(1, count + 1))), encoding="utf-8")
        assert "CARD_COUNT" in _topic_codes(alpha, syllabus), f"{count} cards should fail"


def test_card_ids_must_be_topic_id_hash_number(alpha: Path, syllabus: Syllabus, edit: Edit) -> None:
    edit(alpha / "cards.yaml", "id: tst/alpha#2", "id: tst/alpha#0")
    assert "CARD_ID_FORMAT" in _topic_codes(alpha, syllabus)


def test_card_ids_must_be_unique(alpha: Path, syllabus: Syllabus, edit: Edit) -> None:
    edit(alpha / "cards.yaml", "id: tst/alpha#3", "id: tst/alpha#2")
    assert "CARD_ID_DUPLICATE" in _topic_codes(alpha, syllabus)


# --- interview.md --------------------------------------------------------------------------


def test_interview_file_must_exist(alpha: Path, syllabus: Syllabus) -> None:
    (alpha / "interview.md").unlink()
    assert "INTERVIEW_MISSING" in _topic_codes(alpha, syllabus)


def test_interview_questions_must_be_numbered_in_order(
    alpha: Path, syllabus: Syllabus, edit: Edit
) -> None:
    edit(alpha / "interview.md", "## Q2. Why is mutation", "## Q3. Why is mutation")
    assert "INTERVIEW_QUESTIONS" in _topic_codes(alpha, syllabus)


# --- problems ------------------------------------------------------------------------------


def test_topic_needs_at_least_one_problem(alpha: Path, syllabus: Syllabus) -> None:
    (alpha / "problems").rename(alpha / "problems-off")
    assert "TOPIC_PROBLEM_COUNT" in _topic_codes(alpha, syllabus)


def test_topic_allows_at_most_three_problems(alpha: Path, syllabus: Syllabus) -> None:
    for slug in ("p02-copy", "p03-copy", "p04-copy"):
        shutil.copytree(alpha / "problems" / "p01-double-all", alpha / "problems" / slug)
    assert "TOPIC_PROBLEM_COUNT" in _topic_codes(alpha, syllabus)


def test_problem_directory_must_match_pnn_slug(alpha: Path, syllabus: Syllabus) -> None:
    (alpha / "problems" / "p01-double-all").rename(alpha / "problems" / "P01_Double")
    assert "PROBLEM_DIR_NAME" in _topic_codes(alpha, syllabus)


def test_exactly_one_core_problem(alpha: Path, syllabus: Syllabus, edit: Edit) -> None:
    edit(alpha / "problems" / "p01-double-all" / "meta.yaml", "core: true", "core: false")
    assert "PROBLEM_CORE_COUNT" in _topic_codes(alpha, syllabus)


def test_core_problem_must_be_p01(alpha: Path, syllabus: Syllabus) -> None:
    (alpha / "problems" / "p01-double-all").rename(alpha / "problems" / "p02-double-all")
    assert "PROBLEM_CORE_NOT_P01" in _topic_codes(alpha, syllabus)


def test_problem_meta_id_must_match_its_path(alpha: Path, syllabus: Syllabus, edit: Edit) -> None:
    edit(
        alpha / "problems" / "p01-double-all" / "meta.yaml",
        "id: tst/alpha/p01-double-all",
        "id: tst/alpha/p01-doubled",
    )
    assert "CONTENT_ID_MISMATCH" in _topic_codes(alpha, syllabus)


def test_python_problem_requires_its_files(alpha: Path, syllabus: Syllabus) -> None:
    (alpha / "problems" / "p01-double-all" / "solution.py").unlink()
    assert "PROBLEM_FILE_MISSING" in _topic_codes(alpha, syllabus)


def test_sql_problem_requires_its_files(beta: Path, syllabus: Syllabus) -> None:
    (beta / "problems" / "p01-sql-total-by-user" / "solution.sql").unlink()
    assert "PROBLEM_FILE_MISSING" in _topic_codes(beta, syllabus)


def test_problem_md_needs_the_four_sections_in_order(
    alpha: Path, syllabus: Syllabus, edit: Edit
) -> None:
    edit(alpha / "problems" / "p01-double-all" / "problem.md", "## Constraints", "## Limits")
    assert "PROBLEM_MD_SECTIONS" in _topic_codes(alpha, syllabus)


def test_examples_need_at_least_two_examples(alpha: Path, syllabus: Syllabus, edit: Edit) -> None:
    edit(alpha / "problems" / "p01-double-all" / "problem.md", ">>> double_all([])\n[]\n", "")
    assert "PROBLEM_EXAMPLES" in _topic_codes(alpha, syllabus)


def test_hints_must_be_exactly_three_levels(alpha: Path, syllabus: Syllabus, edit: Edit) -> None:
    edit(alpha / "problems" / "p01-double-all" / "hints.md", "## Hint 3: code shape", "## Sketch")
    assert "HINTS_HEADINGS" in _topic_codes(alpha, syllabus)


def test_python_files_must_parse(alpha: Path, syllabus: Syllabus) -> None:
    (alpha / "problems" / "p01-double-all" / "starter.py").write_text("def double_all(:\n")
    assert "PYTHON_SYNTAX" in _topic_codes(alpha, syllabus)


def test_python_solution_must_define_the_entry(alpha: Path, syllabus: Syllabus, edit: Edit) -> None:
    edit(
        alpha / "problems" / "p01-double-all" / "solution.py",
        "def double_all(values",
        "def doubled(values",
    )
    assert "PROBLEM_ENTRY_MISSING" in _topic_codes(alpha, syllabus)


def test_tests_must_import_the_learner_code_from_submission(
    alpha: Path, syllabus: Syllabus, edit: Edit
) -> None:
    edit(
        alpha / "problems" / "p01-double-all" / "test_public.py",
        "from submission import double_all",
        "import submission",
    )
    assert "PROBLEM_TEST_IMPORT" in _topic_codes(alpha, syllabus)


def test_tests_must_never_import_solution(alpha: Path, syllabus: Syllabus, edit: Edit) -> None:
    edit(
        alpha / "problems" / "p01-double-all" / "test_hidden.py",
        "from submission import double_all",
        "from solution import double_all",
    )
    assert "PROBLEM_TEST_SOLUTION_IMPORT" in _topic_codes(alpha, syllabus)


def test_public_tests_need_at_least_three(alpha: Path, syllabus: Syllabus, edit: Edit) -> None:
    edit(
        alpha / "problems" / "p01-double-all" / "test_public.py",
        "def test_negative_values_stay_negative() -> None:\n"
        "    assert double_all([-3.0]) == [-6.0]\n",
        "",
    )
    assert "PROBLEM_TEST_COUNT" in _topic_codes(alpha, syllabus)


def test_banned_imports_must_be_named_under_constraints(
    alpha: Path, syllabus: Syllabus, edit: Edit
) -> None:
    edit(
        alpha / "problems" / "p01-double-all" / "meta.yaml",
        "banned_imports: []",
        "banned_imports: [numpy.linalg]",
    )
    assert "PROBLEM_BANNED_UNDOCUMENTED" in _topic_codes(alpha, syllabus)


def test_starter_must_be_a_stub(alpha: Path, syllabus: Syllabus) -> None:
    (alpha / "problems" / "p01-double-all" / "starter.py").write_text(
        "def double_all(values: list[float]) -> list[float]:\n    return []\n"
    )
    assert "PROBLEM_STARTER_NOT_STUB" in _topic_codes(alpha, syllabus)


def test_starter_signature_must_match_the_solution(alpha: Path, syllabus: Syllabus) -> None:
    (alpha / "problems" / "p01-double-all" / "starter.py").write_text(
        "def double_all(values: list[float], scale: float) -> list[float]:\n"
        "    raise NotImplementedError\n"
    )
    assert "PROBLEM_SIGNATURE_MISMATCH" in _topic_codes(alpha, syllabus)


def test_sql_tests_must_read_submission_sql(beta: Path, syllabus: Syllabus) -> None:
    tests = beta / "problems" / "p01-sql-total-by-user" / "test_public.py"
    tests.write_text(tests.read_text(encoding="utf-8").replace("submission.sql", "other.sql"))
    assert "PROBLEM_TEST_IMPORT" in _topic_codes(beta, syllabus)


# --- learning paths ------------------------------------------------------------------------


def test_path_topic_must_exist_in_the_syllabus(
    write_core: Callable[[str], CorePath], syllabus: Syllabus
) -> None:
    assert "PATH_TOPIC_NOT_IN_SYLLABUS" in _path_codes(
        "tst/alpha, tst/gamma, tst/nowhere", write_core, syllabus
    )


def test_path_must_not_repeat_a_topic(
    write_core: Callable[[str], CorePath], syllabus: Syllabus
) -> None:
    assert "PATH_DUPLICATE_TOPIC" in _path_codes(
        "tst/alpha, tst/alpha, tst/gamma", write_core, syllabus
    )


def test_every_tier_a_topic_must_be_on_the_path(
    write_core: Callable[[str], CorePath], syllabus: Syllabus
) -> None:
    assert "PATH_MISSING_TIER_A" in _path_codes("tst/alpha", write_core, syllabus)


def test_path_must_not_contain_non_tier_a_topics(
    write_core: Callable[[str], CorePath], syllabus: Syllabus
) -> None:
    assert "PATH_NOT_TIER_A" in _path_codes("tst/alpha, tst/gamma, tst/beta", write_core, syllabus)


# --- scaffold placeholders and resolvable sources --------------------------------------------


def test_scaffold_todo_text_is_an_issue(alpha: Path, syllabus: Syllabus, edit: Edit) -> None:
    hints = alpha / "problems" / "p01-double-all" / "hints.md"
    hints.write_text(hints.read_text() + "\nTODO: the shape of the code.\n")
    assert "TOPIC_TODO" in _topic_codes(alpha, syllabus)


def test_scaffolded_problem_directory_name_is_an_issue(alpha: Path, syllabus: Syllabus) -> None:
    (alpha / "problems" / "p01-double-all").rename(alpha / "problems" / "p01-todo")
    assert "TOPIC_TODO" in _topic_codes(alpha, syllabus)


def _set_sources(topic_dir: Path, line: str) -> None:
    """Replace the single source entry, whatever it currently is."""
    meta = topic_dir / "topic.yaml"
    lines = meta.read_text().splitlines()
    at = lines.index("sources:") + 1
    lines[at] = f'  - "{line}"'
    meta.write_text("\n".join(lines) + "\n")


def test_a_library_source_must_exist(alpha: Path, syllabus: Syllabus) -> None:
    _set_sources(alpha, "library: AI-ML/Missing Note.md")
    assert "SOURCE_NOT_FOUND" in _topic_codes(alpha, syllabus)


def test_a_sources_path_resolves_under_the_repository_sources(
    alpha: Path, syllabus: Syllabus
) -> None:
    note = alpha.parents[2] / "sources" / "interview-prep" / "frameworks" / "RASS.md"
    note.parent.mkdir(parents=True)
    note.write_text("# RASS\n")
    _set_sources(alpha, "source: interview-prep/frameworks/RASS.md")
    assert "SOURCE_NOT_FOUND" not in _topic_codes(alpha, syllabus)
    _set_sources(alpha, "source: interview-prep/frameworks/Missing.md")
    assert "SOURCE_NOT_FOUND" in _topic_codes(alpha, syllabus)
