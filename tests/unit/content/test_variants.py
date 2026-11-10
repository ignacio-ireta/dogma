"""The variants/ manifest: schema, loader and static validation on mutated fixture copies."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from dogma.content import validate
from dogma.content.loader import load_problem, load_topic
from dogma.content.validate import Issue, validate_variants
from dogma.core.errors import ContentError

Edit = Callable[[Path, str, str], None]

MANIFEST_TEXT = """problem_version: 1
accept:
  - file: accept_blind.py
    origin: blind
reject:
  - file: reject_mutates_input.py
    mistake: doubles the caller's list in place
    clause: Must not mutate its input.
    killed_by: test_input_list_is_not_mutated
    visible: hidden
"""


def _codes(issues: list[Issue]) -> set[str]:
    return {issue.code for issue in issues}


def _problem_codes(problem_dir: Path) -> set[str]:
    return _codes(validate_variants(load_problem(problem_dir)))


def test_fixture_as_written_has_no_variant_issues(alpha_problem: Path) -> None:
    assert _problem_codes(alpha_problem) == set()


def test_missing_manifest_is_reported(alpha_problem: Path) -> None:
    (alpha_problem / "variants" / "manifest.yaml").unlink()
    assert _problem_codes(alpha_problem) == {"VARIANT_MANIFEST_MISSING"}


def test_unlisted_variant_file_is_reported(alpha_problem: Path) -> None:
    extra = alpha_problem / "variants" / "reject_extra.py"
    extra.write_text("def double_all(values):\n    return values\n", encoding="utf-8")
    assert _problem_codes(alpha_problem) == {"VARIANT_FILE_UNLISTED"}


def test_listed_file_that_is_gone_is_reported(alpha_problem: Path) -> None:
    (alpha_problem / "variants" / "reject_mutates_input.py").unlink()
    assert _problem_codes(alpha_problem) == {"VARIANT_FILE_MISSING"}


def test_listed_file_with_wrong_suffix_is_reported(alpha_problem: Path, edit: Edit) -> None:
    variants = alpha_problem / "variants"
    (variants / "accept_blind.py").rename(variants / "accept_blind.sql")
    edit(variants / "manifest.yaml", "file: accept_blind.py", "file: accept_blind.sql")
    assert _problem_codes(alpha_problem) == {"VARIANT_SUFFIX"}


def test_clause_changed_by_one_word_is_reported(alpha_problem: Path, edit: Edit) -> None:
    edit(
        alpha_problem / "variants" / "manifest.yaml",
        "clause: Must not mutate its input.",
        "clause: Must not mutate its inputs.",
    )
    assert _problem_codes(alpha_problem) == {"VARIANT_CLAUSE_NOT_IN_STATEMENT"}


def test_killed_by_unknown_test_is_reported(alpha_problem: Path, edit: Edit) -> None:
    edit(
        alpha_problem / "variants" / "manifest.yaml",
        "killed_by: test_input_list_is_not_mutated",
        "killed_by: test_nope",
    )
    assert _problem_codes(alpha_problem) == {"VARIANT_KILLED_BY_UNKNOWN"}


def test_public_visible_with_hidden_test_name_is_reported(alpha_problem: Path, edit: Edit) -> None:
    edit(alpha_problem / "variants" / "manifest.yaml", "visible: hidden", "visible: public")
    assert _problem_codes(alpha_problem) == {"VARIANT_KILLED_BY_UNKNOWN"}


def test_stale_problem_version_is_reported(alpha_problem: Path, edit: Edit) -> None:
    edit(alpha_problem / "variants" / "manifest.yaml", "problem_version: 1", "problem_version: 2")
    assert _problem_codes(alpha_problem) == {"VARIANT_VERSION_STALE"}


def test_clause_differing_only_in_spacing_and_line_breaks_passes(
    alpha_problem: Path, edit: Edit
) -> None:
    edit(
        alpha_problem / "variants" / "manifest.yaml",
        "    clause: Must not mutate its input.",
        "    clause: Must not mutate\n      its   input.",
    )
    assert _problem_codes(alpha_problem) == set()


def test_verified_problem_without_variants_is_reported_when_required(
    alpha_problem: Path, edit: Edit, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validate, "VARIANTS_REQUIRED", True)
    edit(alpha_problem / "meta.yaml", "status: draft", "status: verified")
    shutil.rmtree(alpha_problem / "variants")
    assert _problem_codes(alpha_problem) == {"VARIANTS_REQUIRED"}


def test_verified_problem_with_variants_is_not_reported_when_required(
    alpha_problem: Path, edit: Edit, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validate, "VARIANTS_REQUIRED", True)
    edit(alpha_problem / "meta.yaml", "status: draft", "status: verified")
    assert _problem_codes(alpha_problem) == set()


def test_draft_problem_without_variants_is_not_reported_when_required(
    alpha_problem: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validate, "VARIANTS_REQUIRED", True)
    shutil.rmtree(alpha_problem / "variants")
    assert _problem_codes(alpha_problem) == set()


def test_bad_variant_file_name_fails_the_schema(alpha: Path, alpha_problem: Path) -> None:
    (alpha_problem / "variants" / "manifest.yaml").write_text(
        MANIFEST_TEXT.replace("file: accept_blind.py", "file: Accept-Blind.py"), encoding="utf-8"
    )
    with pytest.raises(ContentError):
        load_topic(alpha)


def test_freedom_origin_without_freedom_fails_the_schema(alpha: Path, alpha_problem: Path) -> None:
    (alpha_problem / "variants" / "manifest.yaml").write_text(
        MANIFEST_TEXT.replace("origin: blind", "origin: freedom"), encoding="utf-8"
    )
    with pytest.raises(ContentError, match="freedom"):
        load_topic(alpha)


def test_two_blind_accepts_fail_the_schema(alpha: Path, alpha_problem: Path) -> None:
    (alpha_problem / "variants" / "accept_blind.sql").write_text("SELECT 1;\n", encoding="utf-8")
    extra = "  - file: accept_blind.sql\n    origin: blind\nreject:\n"
    (alpha_problem / "variants" / "manifest.yaml").write_text(
        MANIFEST_TEXT.replace("reject:\n", extra, 1), encoding="utf-8"
    )
    with pytest.raises(ContentError, match="at most one blind accept"):
        load_topic(alpha)


def test_file_listed_twice_fails_the_schema(alpha: Path, alpha_problem: Path) -> None:
    twice = MANIFEST_TEXT + (
        "  - file: reject_mutates_input.py\n"
        "    mistake: again\n"
        "    clause: Must not mutate its input.\n"
        "    killed_by: test_input_list_is_not_mutated\n"
        "    visible: hidden\n"
    )
    (alpha_problem / "variants" / "manifest.yaml").write_text(twice, encoding="utf-8")
    with pytest.raises(ContentError, match="more than once"):
        load_topic(alpha)


def test_unknown_manifest_key_fails_the_schema(alpha: Path, alpha_problem: Path) -> None:
    (alpha_problem / "variants" / "manifest.yaml").write_text(
        MANIFEST_TEXT + "bogus: 1\n", encoding="utf-8"
    )
    with pytest.raises(ContentError):
        load_topic(alpha)


def test_loader_fills_variants_for_the_fixture(alpha_problem: Path) -> None:
    problem = load_problem(alpha_problem)
    assert problem.variant_files == frozenset(
        {"accept_blind.py", "reject_mutates_input.py", "manifest.yaml"}
    )
    assert problem.variants is not None
    assert [variant.file for variant in problem.variants.accept] == ["accept_blind.py"]


def test_loader_leaves_variants_empty_for_the_beta_fixture(beta: Path) -> None:
    problem = load_problem(beta / "problems" / "p01-sql-total-by-user")
    assert problem.variant_files == frozenset()
    assert problem.variants is None
