"""Allowlisted source discovery and incremental, per-source atomic indexing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import yaml

from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.corpus.chunker import CHUNKER_VERSION, chunk_markdown
from dogma.corpus.index import connect, replace_source, safe_path
from dogma.corpus.models import CorpusIssue, SyncReport

_EXCLUDED = frozenset({"archive", "archives", "runs", "node_modules", "__pycache__"})


@dataclass(frozen=True)
class _Source:
    source_id: str
    path: Path
    topics: tuple[str, ...] | None = None


def _markdown(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    paths: list[Path] = []

    def fail(exc: OSError) -> None:
        raise exc

    for directory, dirs, files in root.walk(on_error=fail):
        dirs[:] = sorted(
            d
            for d in dirs
            if not d.startswith(".") and d not in _EXCLUDED and not (directory / d).is_symlink()
        )
        paths.extend(
            directory / name
            for name in sorted(files)
            if name.lower().endswith(".md")
            and not name.startswith(".")
            and not (directory / name).is_symlink()
        )
    return tuple(paths)


def _mapping(path: Path) -> dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected YAML mapping")
    return data


def _keywords(settings: Settings) -> dict[str, tuple[str, ...]]:
    keywords: dict[str, tuple[str, ...]] = {}
    syllabus = safe_path(settings, "curriculum/syllabus.yaml")
    if syllabus.exists():
        data = _mapping(syllabus)
        tracks = data.get("tracks", [])
        if not isinstance(tracks, list):
            raise ValueError("Expected tracks list")
        for track in tracks:
            if not isinstance(track, dict) or not isinstance(track.get("topics"), list):
                raise ValueError("Expected topics list")
            for topic in track["topics"]:
                if not isinstance(topic, dict):
                    raise ValueError("Expected topic mapping")
                keywords[str(topic["id"])] = (str(topic["title"]).casefold(),)
    return keywords


def _discover(
    settings: Settings, issues: list[CorpusIssue], protected: set[str]
) -> tuple[list[_Source], dict[str, tuple[str, ...]]]:
    sources: list[_Source] = []
    try:
        keywords = _keywords(settings)
    except (OSError, ValueError, yaml.YAMLError, KeyError, TypeError):
        issues.append(_issue("curriculum/syllabus.yaml", "Cannot read topic search keywords."))
        keywords = {}
    for relative, prefix in [("sources", "sources"), ("corpus/md", "corpus")]:
        root = safe_path(settings, relative)
        try:
            for path in _markdown(root):
                rel = path.relative_to(root)
                if len(rel.parts) >= 2:
                    sources.append(_Source(f"{prefix}:{rel.as_posix()}", path))
        except OSError:
            protected.add(prefix + ":")
            issues.append(_issue(relative, "Cannot scan source directory; prior entries retained."))
    content = safe_path(settings, "content")
    if content.exists():
        try:
            _lessons(content, sources, keywords, issues, protected)
        except OSError:
            protected.add("content:")
            issues.append(_issue("content", "Cannot scan content; prior lessons retained."))
    return sources, keywords


def _topic_metadata(content: Path) -> tuple[Path, ...]:
    def fail(exc: OSError) -> None:
        raise exc

    metadata: list[Path] = []
    for directory, dirs, files in content.walk(on_error=fail):
        depth = len(directory.relative_to(content).parts)
        dirs[:] = sorted(
            d
            for d in dirs
            if depth < 2 and not d.startswith(".") and not (directory / d).is_symlink()
        )
        if depth == 2 and "topic.yaml" in files and not (directory / "topic.yaml").is_symlink():
            metadata.append(directory / "topic.yaml")
    return tuple(metadata)


def _lessons(
    content: Path,
    sources: list[_Source],
    keywords: dict[str, tuple[str, ...]],
    issues: list[CorpusIssue],
    protected: set[str],
) -> None:
    # Only fixed-depth topic packages are content sources; exercises never enter retrieval.
    for meta in _topic_metadata(content):
        if any(p.is_symlink() for p in (meta, meta.parent, meta.parent.parent)):
            continue
        topic_id = meta.parent.relative_to(content).as_posix()
        try:
            data = _mapping(meta)
            if data.get("id") != topic_id:
                raise ValueError("Topic id differs from path")
            aliases = data.get("aliases", [])
            if not isinstance(aliases, list) or not all(isinstance(x, str) for x in aliases):
                raise ValueError("Invalid aliases")
            keywords[topic_id] = tuple(
                str(x).casefold() for x in [data.get("title", topic_id), *aliases] if x
            )
            lesson = meta.parent / "lesson.md"
            if data.get("status") in {"verified", "reviewed"} and not lesson.is_symlink():
                sources.append(_Source(f"content:{topic_id}/lesson", lesson, (topic_id,)))
        except (OSError, ValueError, yaml.YAMLError, TypeError):
            protected.add(f"content:{topic_id}/lesson")
            issues.append(
                _issue(
                    f"content/{topic_id}/topic.yaml",
                    "Cannot read topic metadata; prior lesson retained.",
                )
            )


def _issue(path: str, message: str) -> CorpusIssue:
    return CorpusIssue(path, ErrorCode.CORPUS_SOURCE_INVALID, message)


def _index_source(
    conn: sqlite3.Connection,
    source: _Source,
    root: Path,
    keywords: dict[str, tuple[str, ...]],
    previous: dict[str, str],
) -> str:
    path = source.path.relative_to(root).as_posix()
    text = source.path.read_text(encoding="utf-8")
    folded = text.casefold()
    topics = (
        source.topics
        if source.topics is not None
        else tuple(
            sorted(
                topic for topic, terms in keywords.items() if any(term in folded for term in terms)
            )
        )
    )
    digest = hashlib.sha256(
        (CHUNKER_VERSION + "\0" + json.dumps(topics) + "\0" + text).encode("utf-8")
    ).hexdigest()
    if previous.get(source.source_id) == digest:
        return "unchanged"
    chunks = chunk_markdown(text, source_id=source.source_id, path=path, topic_ids=topics)
    replace_source(conn, source.source_id, path, digest, chunks)
    return "updated" if source.source_id in previous else "added"


def sync_index(settings: Settings) -> SyncReport:
    """Scan only sources, approved lessons, and live corpus Markdown; index changes.

    Read/parse failures retain old passages. Only genuinely absent or ineligible
    sources are deleted. A failed directory scan protects its previous entries.
    """
    issues: list[CorpusIssue] = []
    protected: set[str] = set()
    sources, keywords = _discover(settings, issues, protected)
    counts = dict.fromkeys(("added", "updated", "removed", "unchanged"), 0)
    conn: sqlite3.Connection | None = None
    try:
        conn = connect(settings, create=True)
        previous = dict(conn.execute("SELECT source_id,digest FROM sources"))
        seen = {source.source_id for source in sources}
        for source in sources:
            try:
                counts[
                    _index_source(conn, source, settings.repo_root.resolve(), keywords, previous)
                ] += 1
            except (OSError, ValueError, yaml.YAMLError):
                issues.append(
                    _issue(
                        source.path.relative_to(settings.repo_root).as_posix(),
                        "Cannot read or parse source; prior chunks retained.",
                    )
                )
        with conn:
            for source_id in previous.keys() - seen:
                if not any(source_id.startswith(prefix) for prefix in protected):
                    conn.execute("DELETE FROM sources WHERE source_id=?", (source_id,))
                    counts["removed"] += 1
    except (sqlite3.Error, OSError) as exc:
        raise DogmaError(
            ErrorCode.CORPUS_INDEX_INVALID,
            "Cannot update corpus/index.db; check its schema and permissions.",
        ) from exc
    finally:
        if conn is not None:
            conn.close()
    return SyncReport(**counts, issues=tuple(issues))
