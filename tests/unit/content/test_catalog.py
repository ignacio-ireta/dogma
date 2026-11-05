from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dogma.content.catalog import build_catalog, write_catalog
from dogma.core.config import Settings
from dogma.core.errors import ContentError, ErrorCode

NOW = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def repo(tmp_path: Path, fixtures: Path) -> Settings:
    """A throwaway repo laid out like the real one, built from the fixture files."""
    shutil.copytree(fixtures / "packages", tmp_path / "content")
    (tmp_path / "curriculum" / "paths").mkdir(parents=True)
    shutil.copy(fixtures / "syllabus.yaml", tmp_path / "curriculum" / "syllabus.yaml")
    shutil.copy(fixtures / "core.yaml", tmp_path / "curriculum" / "paths" / "core.yaml")
    return Settings(repo_root=tmp_path)


def test_build_catalog_lists_authored_topics_and_their_problems(repo: Settings) -> None:
    catalog = build_catalog(repo, now=NOW)
    assert catalog.version == 1
    assert catalog.generated_at == NOW
    assert [topic.id for topic in catalog.topics] == ["tst/alpha", "tst/beta"]
    alpha, beta = catalog.topics
    assert (alpha.track, alpha.tier, alpha.status) == ("tst", "A", "draft")
    assert [p.id for p in alpha.problems] == ["tst/alpha/p01-double-all"]
    assert alpha.problems[0].language == "python" and alpha.problems[0].minutes == 10
    assert beta.requires == ("tst/alpha",)
    assert beta.problems[0].language == "sql"


def test_build_catalog_refuses_content_that_fails_validation(repo: Settings) -> None:
    topic = repo.content_dir / "tst" / "alpha" / "topic.yaml"
    topic.write_text(topic.read_text().replace("title: Alpha topic", "title: Alpha"))
    with pytest.raises(ContentError) as excinfo:
        build_catalog(repo, now=NOW)
    assert excinfo.value.code == ErrorCode.CONTENT_SCHEMA_INVALID
    assert "TOPIC_TITLE_MISMATCH" in excinfo.value.message


def test_write_catalog_writes_json_atomically(repo: Settings, tmp_path: Path) -> None:
    target = tmp_path / "var" / "catalog.json"
    assert write_catalog(build_catalog(repo, now=NOW), target) is True
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["generated_at"].startswith("2026-10-23T12:00:00")
    assert [t["id"] for t in data["topics"]] == ["tst/alpha", "tst/beta"]
    assert sorted(p.name for p in target.parent.iterdir()) == ["catalog.json"]


def test_write_catalog_is_a_no_op_when_only_the_timestamp_changes(
    repo: Settings, tmp_path: Path
) -> None:
    target = tmp_path / "catalog.json"
    write_catalog(build_catalog(repo, now=NOW), target)
    before = target.read_bytes()
    later = datetime(2026, 11, 1, 9, 0, 0, tzinfo=UTC)
    assert write_catalog(build_catalog(repo, now=later), target) is False
    assert target.read_bytes() == before


def test_write_catalog_rewrites_when_content_changes(repo: Settings, tmp_path: Path) -> None:
    target = tmp_path / "catalog.json"
    write_catalog(build_catalog(repo, now=NOW), target)
    beta = repo.content_dir / "tst" / "beta" / "topic.yaml"
    beta.write_text(beta.read_text().replace("status: draft", "status: verified"))
    assert write_catalog(build_catalog(repo, now=NOW), target) is True
    assert '"status": "verified"' in target.read_text(encoding="utf-8")


def test_write_catalog_replaces_a_corrupt_file(repo: Settings, tmp_path: Path) -> None:
    target = tmp_path / "catalog.json"
    target.write_text("{not json", encoding="utf-8")
    assert write_catalog(build_catalog(repo, now=NOW), target) is True
    assert json.loads(target.read_text(encoding="utf-8"))["version"] == 1
