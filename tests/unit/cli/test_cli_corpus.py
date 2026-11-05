"""Corpus CLI fixture flow remains isolated from learner storage."""

from __future__ import annotations


def test_corpus_cli_sync_search_modify_delete(invoke, repo):
    notes = repo / "sources/notes"
    notes.mkdir(parents=True)
    note = notes / "fixture.md"
    note.write_text("# Orchard\nBananas ripen quickly.")
    first = invoke("corpus", "sync", "--no-convert")
    assert first.exit_code == 0, first.output
    assert "added" in first.output
    second = invoke("corpus", "sync", "--no-convert")
    assert second.exit_code == 0
    assert "unchanged" in second.output
    result = invoke("corpus", "search", "bananas")
    assert result.exit_code == 0, result.output
    assert "sources:notes/fixture.md" in result.output
    assert "Bananas ripen quickly" in result.output
    note.write_text("# Orchard\nKiwifruit ripens slowly.")
    assert invoke("corpus", "sync", "--no-convert").exit_code == 0
    assert "No matches" in invoke("corpus", "search", "bananas").output
    assert "Kiwifruit" in invoke("corpus", "search", "kiwifruit", "--limit", "1").output
    note.unlink()
    assert invoke("corpus", "sync", "--no-convert").exit_code == 0
    assert "No matches" in invoke("corpus", "search", "kiwifruit").output
    assert not (repo / "var").exists()


def test_corpus_cli_missing_index_and_invalid_limit(invoke):
    result = invoke("corpus", "search", "hello")
    assert result.exit_code == 1
    assert "CORPUS_INDEX_MISSING" in result.output
    assert "Traceback" not in result.output
    assert invoke("corpus", "search", "hello", "--limit", "0").exit_code != 0


def test_corpus_sync_issues_exit_nonzero(invoke, repo):
    notes = repo / "sources/notes"
    notes.mkdir(parents=True)
    (notes / "broken.md").write_bytes(b"\xff")
    result = invoke("corpus", "sync", "--no-convert")
    assert result.exit_code == 1
    assert "CORPUS_SOURCE_INVALID" in result.output
    assert "sources/notes/broken.md" in result.output


def test_topic_filter_restricts_cli_hits(invoke):
    assert invoke("corpus", "sync", "--no-convert").exit_code == 0
    result = invoke("corpus", "search", "projection", "--topic", "cml/vector-operations")
    assert result.exit_code == 0
    assert "content:cml/vector-operations/lesson" in result.output
    assert "No matches" in invoke("corpus", "search", "projection", "--topic", "unknown").output
