"""Real FTS fixture exercises sync, literal query handling, and last-good retention."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from dogma.core.config import Settings
from dogma.core.errors import DogmaError
from dogma.corpus import sync
from dogma.corpus.index import search
from dogma.corpus.sync import sync_index

pytestmark = pytest.mark.integration


@pytest.fixture
def corpus_repo(tmp_path: Path) -> Settings:
    (tmp_path / "sources/notes").mkdir(parents=True)
    (tmp_path / "corpus/md/papers").mkdir(parents=True)
    (tmp_path / "curriculum").mkdir()
    (tmp_path / "curriculum/syllabus.yaml").write_text(
        "tracks:\n  - topics:\n      - {id: cml/adam, title: Adam optimizer}\n"
    )
    (tmp_path / "sources/notes/adam.md").write_text("# Adam optimizer\nMomentum adapts gradients.")
    (tmp_path / "sources/notes/banana.md").write_text("# Fruit\nBanana bread with apples.")
    (tmp_path / "corpus/md/papers/adam.pdf.md").write_text(
        "---\nsource_file: adam.pdf\n---\n# Adam\n### Page 3\nAdam optimizer momentum.\n"
    )
    return Settings(repo_root=tmp_path)


def test_expected_hits_topics_and_citations(corpus_repo: Settings) -> None:
    report = sync_index(corpus_repo)
    assert (report.added, report.updated, report.removed, report.unchanged) == (3, 0, 0, 0)
    assert report.issues == ()
    hits = search(corpus_repo, "banana")
    assert len(hits) == 1
    assert hits[0].chunk.source_id == "sources:notes/banana.md"
    assert "sources/notes/banana.md" in hits[0].citation
    hits = search(corpus_repo, "optimizer", topics=("cml/adam",))
    assert {h.chunk.source_id for h in hits} == {
        "sources:notes/adam.md",
        "corpus:papers/adam.pdf.md",
    }
    page_hit = next(h for h in hits if h.chunk.page == 3)
    assert "[corpus:papers/adam.pdf.md p.3]" in page_hit.citation
    assert "(corpus/md/papers/adam.pdf.md)" in page_hit.citation
    assert search(corpus_repo, "banana", topics=("cml/adam",)) == ()
    assert not (corpus_repo.repo_root / "var").exists()


@pytest.mark.parametrize("query", ['"banana"', "banana*", "banana:", "(banana)", "banana';--"])
def test_punctuation_is_literal(corpus_repo: Settings, query: str) -> None:
    sync_index(corpus_repo)
    assert [h.chunk.source_id for h in search(corpus_repo, query)] == ["sources:notes/banana.md"]


def test_operators_cannot_broaden_query(corpus_repo: Settings) -> None:
    sync_index(corpus_repo)
    assert search(corpus_repo, "banana OR optimizer") == ()
    assert search(corpus_repo, '" OR 1=1 --') == ()
    assert search(corpus_repo, "()***:") == ()
    assert search(corpus_repo, "") == ()


@pytest.mark.parametrize("k", [0, -1, 101, True])
def test_limit_is_bounded(corpus_repo: Settings, k: int) -> None:
    with pytest.raises(DogmaError) as exc:
        search(corpus_repo, "banana", k=k)
    assert exc.value.code == "CORPUS_QUERY_INVALID"


def test_incremental_update_delete_and_failed_replacement(corpus_repo: Settings) -> None:
    sync_index(corpus_repo)
    db = corpus_repo.repo_root / "corpus/index.db"
    with sqlite3.connect(db) as conn:
        before = conn.execute("SELECT * FROM sources ORDER BY source_id").fetchall()
    report = sync_index(corpus_repo)
    assert (report.added, report.updated, report.unchanged) == (0, 0, 3)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT * FROM sources ORDER BY source_id").fetchall() == before
    note = corpus_repo.repo_root / "sources/notes/banana.md"
    note.write_text("# Fruit\nKiwifruit.")
    (corpus_repo.repo_root / "sources/notes/adam.md").unlink()
    report = sync_index(corpus_repo)
    assert (report.updated, report.removed, report.unchanged) == (1, 1, 1)
    assert search(corpus_repo, "banana") == ()
    assert len(search(corpus_repo, "kiwifruit")) == 1
    note.write_bytes(b"\xff\xfe")
    report = sync_index(corpus_repo)
    assert report.issues[0].path == "sources/notes/banana.md"
    assert len(search(corpus_repo, "kiwifruit")) == 1
    note.write_text("---\nbroken front matter")
    assert sync_index(corpus_repo).issues
    assert len(search(corpus_repo, "kiwifruit")) == 1


def test_excludes_archives_logs_drafts_and_symlinks(corpus_repo: Settings, tmp_path: Path) -> None:
    root = corpus_repo.repo_root
    for rel in [
        "corpus/md/.docmd/archive/old.md",
        "corpus/md/runs/log.md",
        "sources/notes/archive/old.md",
        "content/cml/draft/lesson.md",
    ]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Forbiddenword")
    (root / "content/cml/draft/topic.yaml").write_text("id: cml/draft\nstatus: draft\n")
    outside = tmp_path.parent / (tmp_path.name + "-outside.md")
    outside.write_text("Forbiddenword")
    (root / "sources/notes/escape.md").symlink_to(outside)
    (root / "sources/notes/escape-dir").symlink_to(outside.parent, target_is_directory=True)
    assert sync_index(corpus_repo).added == 3
    assert search(corpus_repo, "Forbiddenword") == ()


def test_reviewed_lesson_alias_tag_and_metadata_change(corpus_repo: Settings) -> None:
    root = corpus_repo.repo_root
    topic = root / "content/cml/adam"
    topic.mkdir(parents=True)
    meta = topic / "topic.yaml"
    meta.write_text("id: cml/adam\ntitle: Adam optimizer\nstatus: reviewed\naliases: [bread]\n")
    (topic / "lesson.md").write_text("# Learn\nAdaptive gradient algorithm.")
    sync_index(corpus_repo)
    assert len(search(corpus_repo, "bread", topics=("cml/adam",))) == 1
    assert search(corpus_repo, "algorithm")[0].chunk.source_id == "content:cml/adam/lesson"
    meta.write_text("id: cml/adam\ntitle: Adam optimizer\nstatus: retired\naliases: []\n")
    report = sync_index(corpus_repo)
    assert report.removed == 1
    assert search(corpus_repo, "algorithm") == ()
    assert search(corpus_repo, "bread", topics=("cml/adam",)) == ()


@pytest.mark.parametrize("corrupt", [False, True])
def test_missing_or_corrupt_index_is_domain_error(tmp_path: Path, corrupt: bool) -> None:
    if corrupt:
        (tmp_path / "corpus").mkdir()
        (tmp_path / "corpus/index.db").write_text("not sqlite")
    with pytest.raises(DogmaError) as exc:
        search(Settings(repo_root=tmp_path), "adam")
    assert exc.value.code == ("CORPUS_INDEX_INVALID" if corrupt else "CORPUS_INDEX_MISSING")


def test_index_symlink_cannot_write_outside_repo(corpus_repo: Settings, tmp_path: Path) -> None:
    outside = tmp_path.parent / (tmp_path.name + "-index")
    outside.write_text("preserve")
    (corpus_repo.repo_root / "corpus/index.db").symlink_to(outside)
    with pytest.raises(DogmaError) as exc:
        sync_index(corpus_repo)
    assert exc.value.code == "CORPUS_PATH_INVALID"
    assert outside.read_text() == "preserve"


def test_broken_fts_schema_is_rejected_even_when_sources_unchanged(corpus_repo: Settings) -> None:
    sync_index(corpus_repo)
    with sqlite3.connect(corpus_repo.repo_root / "corpus/index.db") as conn:
        conn.execute("DROP TABLE chunks_fts")
    with pytest.raises(DogmaError) as exc:
        sync_index(corpus_repo)
    assert exc.value.code == "CORPUS_INDEX_INVALID"


def test_failed_directory_scan_retains_sources(
    corpus_repo: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_index(corpus_repo)
    scan = sync._markdown

    def unreadable(root: Path) -> tuple[Path, ...]:
        if root.name == "sources":
            raise PermissionError("fixture")
        return scan(root)

    monkeypatch.setattr(sync, "_markdown", unreadable)
    report = sync_index(corpus_repo)
    assert report.removed == 0
    assert report.issues
    assert len(search(corpus_repo, "banana")) == 1


def test_sqlite_failure_rolls_back_source_replacement(corpus_repo: Settings) -> None:
    sync_index(corpus_repo)
    with sqlite3.connect(corpus_repo.repo_root / "corpus/index.db") as conn:
        conn.executescript("""
        CREATE TRIGGER fail_replacement BEFORE INSERT ON chunks
        WHEN new.text = 'Invalid replacement'
        BEGIN SELECT RAISE(ABORT,'fixture'); END;
        """)
    (corpus_repo.repo_root / "sources/notes/banana.md").write_text("Invalid replacement")
    with pytest.raises(DogmaError):
        sync_index(corpus_repo)
    assert len(search(corpus_repo, "banana")) == 1


def test_oversized_query_rejected(corpus_repo: Settings) -> None:
    with pytest.raises(DogmaError) as exc:
        search(corpus_repo, "a" * 4097)
    assert exc.value.code == "CORPUS_QUERY_INVALID"


def test_stable_bm25_tie_break(corpus_repo: Settings) -> None:
    root = corpus_repo.repo_root / "sources/notes"
    (root / "z.md").write_text("Uniquequasar")
    (root / "a.md").write_text("Uniquequasar")
    sync_index(corpus_repo)
    hits = search(corpus_repo, "uniquequasar")
    assert [h.chunk.source_id for h in hits] == ["sources:notes/a.md", "sources:notes/z.md"]
    assert hits[0].score == hits[1].score


def test_unreadable_content_track_preserves_last_good_lesson(corpus_repo: Settings) -> None:
    track = corpus_repo.repo_root / "content/cml"
    topic = track / "adam"
    topic.mkdir(parents=True)
    (topic / "topic.yaml").write_text("id: cml/adam\nstatus: verified\naliases: []")
    (topic / "lesson.md").write_text("Uniqueproof")
    sync_index(corpus_repo)
    track.chmod(0)
    try:
        report = sync_index(corpus_repo)
    finally:
        track.chmod(0o755)
    assert report.removed == 0
    assert report.issues
    assert len(search(corpus_repo, "uniqueproof")) == 1


def test_citation_escapes_markdown_filename_punctuation(corpus_repo: Settings) -> None:
    path = corpus_repo.repo_root / "sources/notes/paper [v1] (draft).md"
    path.write_text("Uniquephoton")
    sync_index(corpus_repo)
    hit = search(corpus_repo, "uniquephoton")[0]
    assert hit.chunk.source_id == "sources:notes/paper [v1] (draft).md"
    assert hit.citation == (
        "[sources:notes/paper \\[v1\\] (draft).md](sources/notes/paper%20%5Bv1%5D%20%28draft%29.md)"
    )
