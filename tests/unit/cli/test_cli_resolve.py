"""Id and path resolution: what the learner can type, and what each form maps to."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.unit.cli.support import PROBLEM

from dogma.content.resolve import (
    resolve_problem,
    resolve_target,
    set_aside_workspace,
    workspace_file,
)
from dogma.core.errors import ContentError


def test_problem_id_resolves_to_its_content_directory(settings, problem_dir):
    ref = resolve_problem(settings, PROBLEM)

    assert ref.dir == problem_dir.resolve()
    assert ref.id == PROBLEM
    assert ref.meta.language == "python"
    assert ref.suffix == ".py"


def test_problem_directory_path_resolves_to_the_same_id(settings, problem_dir):
    assert resolve_problem(settings, str(problem_dir)).id == PROBLEM


def test_relative_problem_path_resolves_from_the_working_directory(
    settings, repo: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(repo)
    ref = resolve_problem(settings, "content/cml/vector-operations/problems/p01-vector-projection")
    assert ref.id == PROBLEM


def test_workspace_file_matches_the_scripts_grade_layout(settings):
    ref = resolve_problem(settings, PROBLEM)
    assert workspace_file(settings, ref) == (
        settings.repo_root / "workspace" / "cml" / "vector-operations" / "p01-vector-projection.py"
    )


@pytest.mark.parametrize("ref", ["cml/vector-operations/p09-missing", "nope", "a/b/c/d", ""])
def test_unknown_problem_refs_are_content_not_found(settings, ref):
    with pytest.raises(ContentError) as caught:
        resolve_problem(settings, ref)
    assert caught.value.code.value == "CONTENT_NOT_FOUND"


def test_a_topic_directory_is_not_a_problem(settings, repo):
    with pytest.raises(ContentError, match="not a problem directory"):
        resolve_problem(settings, str(repo / "content" / "cml" / "vector-operations"))


def test_a_directory_outside_content_is_rejected(settings, tmp_path):
    outside = tmp_path / "elsewhere" / "problems" / "p01-x"
    outside.mkdir(parents=True)
    with pytest.raises(ContentError, match="not a problem directory"):
        resolve_problem(settings, str(outside))


def test_topic_id_and_problem_id_both_resolve_to_a_topic(settings, repo):
    topic_dir = repo / "content" / "cml" / "vector-operations"
    by_topic = resolve_target(settings, "cml/vector-operations")
    by_problem = resolve_target(settings, PROBLEM)

    assert by_topic.topic_dir == topic_dir.resolve()
    assert by_topic.problem_dir is None
    assert by_problem.topic_dir == topic_dir.resolve()
    assert by_problem.problem_dir is not None
    assert by_problem.problem_dir.name == "p01-vector-projection"


def test_unknown_target_is_content_not_found(settings):
    with pytest.raises(ContentError, match="no topic or problem"):
        resolve_target(settings, "cml/does-not-exist")


def test_set_aside_copies_the_workspace_file_into_history_and_leaves_it_in_place(settings):
    ref = resolve_problem(settings, PROBLEM)
    source = workspace_file(settings, ref)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("my work\n")

    kept = set_aside_workspace(settings, ref, at=datetime(2026, 10, 26, 9, 30, tzinfo=UTC))

    assert kept == (
        settings.var_dir
        / "workspace-history/cml/vector-operations/p01-vector-projection.20261026T093000000000Z.py"
    )
    assert kept.read_text() == "my work\n"
    assert source.read_text() == "my work\n", "the caller replaces it, not set_aside"


def test_set_aside_without_a_workspace_file_keeps_nothing(settings):
    ref = resolve_problem(settings, PROBLEM)

    assert set_aside_workspace(settings, ref, at=datetime.now(UTC)) is None
    assert not (settings.var_dir / "workspace-history").exists()
