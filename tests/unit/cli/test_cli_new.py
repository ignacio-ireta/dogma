"""``dogma content new``: scaffold into a temporary content tree, refuse what it must refuse."""

from __future__ import annotations

import shutil
from pathlib import Path

from dogma.content.loader import load_syllabus, load_topic
from dogma.content.validate import validate_topic

NEW_TOPIC = "cml/cosine-similarity"
PROBLEM_FILES = (
    "problem.md",
    "meta.yaml",
    "starter.py",
    "solution.py",
    "test_public.py",
    "test_hidden.py",
    "hints.md",
)


def test_new_scaffolds_every_required_file(invoke, settings):
    outcome = invoke("content", "new", NEW_TOPIC)

    target = settings.content_dir / NEW_TOPIC
    assert outcome.exit_code == 0, outcome.output
    for name in ("topic.yaml", "lesson.md", "cards.yaml", "interview.md"):
        assert (target / name).is_file(), name
    for name in PROBLEM_FILES:
        assert (target / "problems" / "p01-todo" / name).is_file(), name


def test_new_fills_id_title_and_tier_from_the_syllabus(invoke, settings):
    invoke("content", "new", NEW_TOPIC)

    syllabus = load_syllabus(settings.syllabus_path)
    entry = syllabus.topics_by_id()[NEW_TOPIC]
    package = load_topic(settings.content_dir / NEW_TOPIC)
    assert package.meta.id == NEW_TOPIC
    assert package.meta.title == entry.title
    assert package.meta.tier == entry.tier
    assert package.lesson is not None


def test_no_template_placeholder_is_left_unfilled(invoke, settings):
    invoke("content", "new", NEW_TOPIC)

    for path in (settings.content_dir / NEW_TOPIC).rglob("*"):
        if path.is_file():
            assert "{{TOPIC_ID}}" not in path.read_text(encoding="utf-8"), path
            assert "{{TITLE" not in path.read_text(encoding="utf-8"), path


def test_scaffold_is_structurally_valid_and_loads(invoke, settings):
    # Placeholders pass every structural rule on purpose; the only issues a fresh scaffold has
    # are its TODO placeholders, so it cannot be marked verified until they are filled in.
    invoke("content", "new", NEW_TOPIC)

    package = load_topic(settings.content_dir / NEW_TOPIC)
    syllabus = load_syllabus(settings.syllabus_path)
    codes = {issue.code for issue in validate_topic(package, syllabus)}
    assert codes == {"TOPIC_TODO"}


def test_new_refuses_an_existing_topic(invoke, settings):
    invoke("content", "new", NEW_TOPIC)
    outcome = invoke("content", "new", NEW_TOPIC)

    assert outcome.exit_code == 2
    assert "already exists; not overwritten" in outcome.output


def test_new_refuses_an_id_that_is_not_in_the_syllabus(invoke, settings):
    outcome = invoke("content", "new", "cml/not-in-the-syllabus")

    assert outcome.exit_code == 2
    assert "not in curriculum/syllabus.yaml" in outcome.output
    assert not (settings.content_dir / "cml" / "not-in-the-syllabus").exists()


def test_new_needs_the_templates_directory(invoke, repo: Path):
    shutil.rmtree(repo / "templates")
    outcome = invoke("content", "new", NEW_TOPIC)

    assert outcome.exit_code == 1
    assert "error [CONTENT_NOT_FOUND]" in outcome.output
    assert not (repo / "content" / "cml" / "cosine-similarity").exists()


CASE_ID = "sys/churn-prediction-case"
CASE_FILES = (
    "topic.yaml",
    "lesson.md",
    "cards.yaml",
    "brief.md",
    "reveals.yaml",
    "twist.md",
    "debrief.md",
)


def test_new_scaffolds_a_case_for_a_sys_case_id(invoke, settings):
    outcome = invoke("content", "new", CASE_ID)

    target = settings.content_dir / CASE_ID
    assert outcome.exit_code == 0, outcome.output
    for name in CASE_FILES:
        assert (target / name).is_file(), name
    assert not (target / "problems").exists()
    package = load_topic(target)
    assert package.meta.kind == "case"
    assert package.case is not None


def test_case_scaffold_fails_verification_only_on_todos(invoke, settings):
    invoke("content", "new", CASE_ID)

    package = load_topic(settings.content_dir / CASE_ID)
    syllabus = load_syllabus(settings.syllabus_path)
    assert {issue.code for issue in validate_topic(package, syllabus)} == {"CASE_TODO"}


def test_new_topic_id_still_scaffolds_a_topic(invoke, settings):
    invoke("content", "new", NEW_TOPIC)

    target = settings.content_dir / NEW_TOPIC
    assert (target / "problems" / "p01-todo" / "problem.md").is_file()
    assert load_topic(target).meta.kind == "topic"
