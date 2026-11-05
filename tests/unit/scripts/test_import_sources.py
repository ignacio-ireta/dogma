"""Tests for scripts/import_sources.py: glob semantics, rules, renames and scrub-preserving sync.

Every test runs against a temporary repo; the real origin roots and sources/ are never touched.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "import_sources.py"
REPO_ROOT = SCRIPT.parents[1]
MANIFEST = REPO_ROOT / "sources" / "manifest.yaml"
REFRESH_INVENTORY = REPO_ROOT / "docs" / "source-refresh-2026-10-28.json"


def load_importer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("import_sources", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


imp = load_importer()


@pytest.mark.parametrize(
    ("pattern", "rel", "expected"),
    [
        ("**/*.md", "a.md", True),
        ("**/*.md", "x/y/z.md", True),
        ("AI-ML/**/*.md", "AI-ML/b/c.md", True),
        ("AI-ML/**/*.md", "AI-MLX/a.md", False),
        ("*.ipynb", "a/b.ipynb", False),
        ("*.ipynb", "b.ipynb", True),
        ("lessons/ref-*.md", "lessons/ref-x.md", True),
        ("lessons/ref-*.md", "lessons/x/ref-y.md", False),
        ("src/**", "src/pkg/mod.py", True),
        ("**/__pycache__/**", "a/__pycache__/m.pyc", True),
        ("drills/**/*", "drills/sql/01.md", True),
        ("file?.txt", "file1.txt", True),
        ("file?.txt", "file12.txt", False),
    ],
)
def test_glob_semantics(pattern: str, rel: str, expected: bool) -> None:
    assert bool(imp.glob_to_regex(pattern).fullmatch(rel)) is expected


def test_last_matching_rule_wins() -> None:
    rules = [
        (imp.glob_to_regex("**/*.md"), "import"),
        (imp.glob_to_regex("drafts/**"), "exclude"),
    ]
    assert imp.is_selected("notes/a.md", rules)
    assert not imp.is_selected("drafts/a.md", rules)
    assert not imp.is_selected("notes/a.csv", rules)


def test_real_manifest_imports_only_reviewed_reference_lessons() -> None:
    manifest = yaml.safe_load((SCRIPT.parents[1] / "sources/manifest.yaml").read_text())
    snapshot = next(item for item in manifest["snapshots"] if item["id"] == "interview-prep")
    rules = [(imp.glob_to_regex(rule["glob"]), rule["action"]) for rule in snapshot["rules"]]
    assert imp.is_selected("lessons/ref-model-selection.md", rules)
    assert not imp.is_selected("lessons/ref-personal-story.md", rules)
    assert not imp.is_selected("lessons/ref-unreviewed-note.md", rules)


def test_rename_changes_basename_only() -> None:
    renames = [(re.compile(r"^(.* Interview Bridge) - .*\.md$"), r"\1.md")]
    assert imp.apply_renames("dir/A Interview Bridge - Context.md", renames) == (
        "dir/A Interview Bridge.md"
    )
    assert imp.apply_renames("plain.md", renames) == "plain.md"


@pytest.fixture
def temp_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    (repo / "sources").mkdir(parents=True)
    monkeypatch.setattr(imp, "REPO", repo)
    monkeypatch.setattr(imp, "LOCK_PATH", repo / "sources" / "manifest.lock.json")
    monkeypatch.setattr(imp, "WITHDRAWN_PATH", repo / "sources" / "withdrawn.yaml")
    return repo


def make_item(origin: Path) -> object:
    return imp.PlannedFile(
        source_id="s",
        origin_path=origin,
        origin_rel="a.md",
        dest_rel="sources/s/a.md",
        origin_commit=None,
    )


def test_sync_is_idempotent_and_preserves_scrubbed_copies(tmp_path: Path, temp_repo: Path) -> None:
    origin = tmp_path / "origin" / "a.md"
    origin.parent.mkdir()
    origin.write_text("one\n", encoding="utf-8")
    item = make_item(origin)
    dest = temp_repo / "sources" / "s" / "a.md"
    hashes = {"sources/s/a.md": imp.sha256_of(origin)}

    entries, counts = imp.sync([item], {}, set(), hashes, force=False)
    assert counts["s"] == {"copied": 1}
    assert imp.write_lock(entries) is True
    assert imp.write_lock(imp.load_lock()) is False  # unchanged re-write is a no-op

    entries, counts = imp.sync([item], imp.load_lock(), set(), hashes, force=False)
    assert counts["s"] == {"kept": 1}
    assert imp.write_lock(entries) is False

    dest.write_text("scrubbed\n", encoding="utf-8")
    relocked = imp.relock([item], imp.load_lock(), hashes)
    assert relocked["sources/s/a.md"]["scrubbed"] is True
    imp.write_lock(relocked)

    entries, counts = imp.sync([item], imp.load_lock(), set(), hashes, force=False)
    assert counts["s"] == {"kept": 1}
    assert dest.read_text(encoding="utf-8") == "scrubbed\n"


def test_changed_origin_is_skipped_unless_forced(tmp_path: Path, temp_repo: Path) -> None:
    origin = tmp_path / "origin" / "a.md"
    origin.parent.mkdir()
    origin.write_text("one\n", encoding="utf-8")
    item = make_item(origin)
    dest = temp_repo / "sources" / "s" / "a.md"
    entries, _ = imp.sync([item], {}, set(), {"sources/s/a.md": imp.sha256_of(origin)}, False)
    imp.write_lock(entries)

    origin.write_text("two\n", encoding="utf-8")
    changed = {"sources/s/a.md": imp.sha256_of(origin)}
    _, counts = imp.sync([item], imp.load_lock(), set(), changed, force=False)
    assert counts["s"] == {"skipped": 1}
    assert dest.read_text(encoding="utf-8") == "one\n"

    _, counts = imp.sync([item], imp.load_lock(), set(), changed, force=True)
    assert counts["s"] == {"forced": 1}
    assert dest.read_text(encoding="utf-8") == "two\n"


def test_withdrawn_destination_is_never_copied(tmp_path: Path, temp_repo: Path) -> None:
    origin = tmp_path / "origin" / "a.md"
    origin.parent.mkdir()
    origin.write_text("one\n", encoding="utf-8")
    item = make_item(origin)
    hashes = {"sources/s/a.md": imp.sha256_of(origin)}
    entries, counts = imp.sync([item], {}, {"sources/s/a.md"}, hashes, force=False)
    assert counts["s"] == {"withdrawn": 1}
    assert entries == {}
    assert not (temp_repo / "sources" / "s" / "a.md").exists()


# --- Refresh options: --source-id, --kind, corpus include, --inventory ---------------------------


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def written_files(repo: Path) -> list[str]:
    return sorted(p.relative_to(repo).as_posix() for p in repo.rglob("*") if p.is_file())


def file_state(repo: Path) -> dict[str, tuple[bytes, int]]:
    """Bytes and mtime of every file, so a rerun that rewrites anything is detected."""
    return {
        p.relative_to(repo).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in repo.rglob("*")
        if p.is_file()
    }


BASE_MANIFEST: dict[str, Any] = {
    "version": 2,
    "snapshots": [
        {
            "id": "notes",
            "root": "library",
            "rules": [
                {"glob": "notes/a.md", "action": "import"},
                {"glob": "notes/b.md", "action": "import"},
            ],
        },
        {
            "id": "elsewhere",
            "root": "unrelated",
            "rules": [{"glob": "**/*.md", "action": "import"}],
        },
    ],
    "corpus": [{"id": "papers", "root": "library", "include": ["papers/p1.pdf"]}],
    "vendor": [],
}


@dataclass(frozen=True)
class Env:
    repo: Path
    origin: Path
    manifest: Path
    roots: Path
    tmp: Path

    def write_manifest(self, data: dict[str, Any]) -> None:
        write_text(self.manifest, yaml.safe_dump(data, sort_keys=False))

    def write_inventory(self, items: list[tuple[str, str]]) -> Path:
        path = self.tmp / "inventory.json"
        payload = {"version": 1, "items": [{"path": p, "sha256": s} for p, s in items]}
        write_text(path, json.dumps(payload))
        return path

    def sha(self, rel: str) -> str:
        return imp.sha256_of(self.origin / rel)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    repo = tmp_path / "repo"
    (repo / "sources").mkdir(parents=True)
    origin = tmp_path / "origin" / "library"
    write_text(origin / "notes" / "a.md", "alpha\n")
    write_text(origin / "notes" / "b.md", "beta\n")
    write_text(origin / "papers" / "p1.pdf", "%PDF one\n")
    write_text(origin / "papers" / "p2.pdf", "%PDF two\n")
    result = Env(repo, origin, tmp_path / "manifest.yaml", tmp_path / "roots.yaml", tmp_path)
    result.write_manifest(BASE_MANIFEST)
    # Deliberately no "unrelated" key: any run that resolves that entry fails.
    write_text(result.roots, yaml.safe_dump({"library": str(origin)}))
    monkeypatch.setattr(imp, "REPO", repo)
    monkeypatch.setattr(imp, "LOCK_PATH", repo / "sources" / "manifest.lock.json")
    monkeypatch.setattr(imp, "WITHDRAWN_PATH", repo / "sources" / "withdrawn.yaml")
    monkeypatch.setattr(imp, "MANIFEST_PATH", result.manifest)
    monkeypatch.setattr(imp, "ROOTS_PATH", result.roots)
    return result


def test_source_id_selector_never_resolves_unrelated_roots(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    walked: list[Path] = []
    original_walk = imp.walk_files

    def spy(root: Path) -> list[str]:
        walked.append(root)
        return original_walk(root)

    monkeypatch.setattr(imp, "walk_files", spy)
    assert imp.main(["--source-id", "notes"]) == 0
    assert walked == [env.origin]
    assert (env.repo / "sources" / "notes" / "notes" / "a.md").read_text() == "alpha\n"


def test_full_run_still_fails_on_missing_root(env: Env) -> None:
    with pytest.raises(SystemExit, match="unrelated"):
        imp.main([])
    assert written_files(env.repo) == []


def test_unknown_source_id_is_rejected(env: Env) -> None:
    with pytest.raises(SystemExit, match="unknown"):
        imp.main(["--source-id", "nope"])
    assert written_files(env.repo) == []


def test_corpus_include_copies_only_listed_files(env: Env) -> None:
    assert imp.main(["--source-id", "papers"]) == 0
    # The origin's relative path is kept under corpus/raw/<id>/, as it is for glob entries.
    assert written_files(env.repo) == [
        "corpus/raw/papers/papers/p1.pdf",
        "sources/manifest.lock.json",
    ]


def test_missing_include_file_stops_before_any_copy(env: Env) -> None:
    corpus = [{"id": "papers", "root": "library", "include": ["papers/p1.pdf", "papers/gone.pdf"]}]
    env.write_manifest({**BASE_MANIFEST, "corpus": corpus})
    (env.origin / "papers" / "p1.pdf").unlink()
    with pytest.raises(SystemExit) as exc:
        imp.main(["--source-id", "papers"])
    message = str(exc.value.code)
    assert "papers/gone.pdf" in message
    assert str(env.origin) not in message
    assert written_files(env.repo) == []


def test_corpus_rejects_glob_together_with_include(env: Env) -> None:
    corpus = [{"id": "papers", "root": "library", "glob": "*.pdf", "include": ["papers/p1.pdf"]}]
    env.write_manifest({**BASE_MANIFEST, "corpus": corpus})
    with pytest.raises(SystemExit):
        imp.main(["--source-id", "papers"])
    assert written_files(env.repo) == []


def test_kind_filter_reports_excluded_corpus_entry(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    assert imp.main(["--source-id", "notes", "--source-id", "papers", "--kind", "snapshot"]) == 0
    assert "not requested: corpus papers" in capsys.readouterr().out
    assert written_files(env.repo) == [
        "sources/manifest.lock.json",
        "sources/notes/notes/a.md",
        "sources/notes/notes/b.md",
    ]


def test_validate_inventory_checks_only_selected_items(env: Env) -> None:
    inventory = env.write_inventory(
        [("notes/a.md", env.sha("notes/a.md")), ("notes/b.md", "0" * 64)]
    )
    imp.validate_inventory(env.origin, inventory, selected=lambda rel: rel == "notes/a.md")
    with pytest.raises(SystemExit, match=re.escape("notes/b.md")):
        imp.validate_inventory(env.origin, inventory, selected=lambda rel: True)


def test_inventory_mismatch_writes_nothing_and_reports_relative_paths(env: Env) -> None:
    inventory = env.write_inventory(
        [("notes/a.md", env.sha("notes/a.md")), ("notes/b.md", "0" * 64)]
    )
    with pytest.raises(SystemExit) as exc:
        imp.main(["--source-id", "notes", "--inventory", str(inventory)])
    message = str(exc.value.code)
    assert "notes/b.md" in message
    assert "notes/a.md" not in message
    assert str(env.origin) not in message
    assert written_files(env.repo) == []


def test_inventory_missing_file_is_counted_and_reported_by_relative_path(env: Env) -> None:
    (env.origin / "papers" / "p1.pdf").unlink()
    inventory = env.write_inventory([("papers/p1.pdf", "f" * 64)])
    with pytest.raises(SystemExit) as exc:
        imp.main(["--source-id", "papers", "--inventory", str(inventory)])
    message = str(exc.value.code)
    assert "1 selected file(s) missing" in message
    assert "papers/p1.pdf" in message
    assert str(env.origin) not in message
    assert written_files(env.repo) == []


def test_inventory_match_imports_selected_notes(env: Env) -> None:
    inventory = env.write_inventory(
        [("notes/a.md", env.sha("notes/a.md")), ("notes/b.md", env.sha("notes/b.md"))]
    )
    assert imp.main(["--source-id", "notes", "--inventory", str(inventory)]) == 0
    assert (env.repo / "sources" / "notes" / "notes" / "b.md").read_text() == "beta\n"


def test_rerun_with_unchanged_origins_writes_nothing(env: Env) -> None:
    assert imp.main(["--source-id", "notes"]) == 0
    before = file_state(env.repo)
    assert imp.main(["--source-id", "notes"]) == 0
    assert file_state(env.repo) == before


def test_scrubbed_copy_survives_changed_origin_without_force(env: Env) -> None:
    assert imp.main(["--source-id", "notes"]) == 0
    dest = env.repo / "sources" / "notes" / "notes" / "a.md"
    dest.write_text("scrubbed\n", encoding="utf-8")
    assert imp.main(["--relock", "--source-id", "notes"]) == 0
    write_text(env.origin / "notes" / "a.md", "changed origin\n")
    assert imp.main(["--source-id", "notes"]) == 0
    assert dest.read_text(encoding="utf-8") == "scrubbed\n"
    lock = json.loads(imp.LOCK_PATH.read_text(encoding="utf-8"))
    entry = next(e for e in lock if e["dest"] == "sources/notes/notes/a.md")
    assert entry["scrubbed"] is True


def test_scoped_relock_keeps_other_sources_lock_entries(env: Env) -> None:
    assert imp.main(["--source-id", "notes"]) == 0
    lock = imp.load_lock()
    lock["corpus/raw/papers/p1.pdf"] = {
        "dest": "corpus/raw/papers/p1.pdf",
        "source_id": "papers",
        "origin_rel": "papers/p1.pdf",
        "origin_commit": None,
        "sha256_origin": "a" * 64,
        "sha256": "a" * 64,
        "scrubbed": False,
    }
    imp.write_lock(lock)
    assert imp.main(["--relock", "--source-id", "notes"]) == 0
    dests = {e["dest"] for e in json.loads(imp.LOCK_PATH.read_text(encoding="utf-8"))}
    assert "corpus/raw/papers/p1.pdf" in dests


def test_real_manifest_library_refresh_matches_inventory() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    items = json.loads(REFRESH_INVENTORY.read_text(encoding="utf-8"))["items"]
    notes = sorted(i["path"] for i in items if i["path"].endswith(".md"))
    pdfs = sorted(i["path"] for i in items if i["path"].endswith(".pdf"))
    assert (len(notes), len(pdfs)) == (36, 72)

    snapshot = next(s for s in manifest["snapshots"] if s["id"] == "library-refresh")
    assert snapshot["root"] == "library"
    assert sorted(rule["glob"] for rule in snapshot["rules"]) == notes
    assert {rule["action"] for rule in snapshot["rules"]} == {"import"}

    corpus = next(c for c in manifest["corpus"] if c["id"] == "library-refresh")
    assert corpus["root"] == "library"
    assert "glob" not in corpus
    assert sorted(corpus["include"]) == pdfs

    text = MANIFEST.read_text(encoding="utf-8")
    assert "/home/" not in text
    assert "/Users/" not in text
