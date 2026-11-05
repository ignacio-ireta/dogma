"""Separate SQLite FTS5 corpus database; it never stores learner state."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from urllib.parse import quote

from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.corpus.models import Chunk, SearchHit

SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE sources (source_id TEXT PRIMARY KEY, path TEXT NOT NULL, digest TEXT NOT NULL);
CREATE TABLE chunks (id INTEGER PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(source_id)
 ON DELETE CASCADE, path TEXT NOT NULL, page INTEGER, headings TEXT NOT NULL,
 ordinal INTEGER NOT NULL, text TEXT NOT NULL, topic_ids TEXT NOT NULL);
CREATE TABLE chunk_topics (chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
 topic_id TEXT NOT NULL, PRIMARY KEY(chunk_id, topic_id));
CREATE VIRTUAL TABLE chunks_fts USING fts5(text, headings, content='chunks', content_rowid='id',
 tokenize='porter unicode61');
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
 INSERT INTO chunks_fts(rowid,text,headings) VALUES(new.id,new.text,new.headings); END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
 INSERT INTO chunks_fts(chunks_fts,rowid,text,headings)
 VALUES('delete',old.id,old.text,old.headings); END;
PRAGMA user_version=1;
"""


def safe_path(settings: Settings, relative: str) -> Path:
    """Reject symlinked paths before reading or writing corpus-controlled locations."""
    root = settings.repo_root.resolve()
    path = root / relative
    current = root
    for part in Path(relative).parts:
        current /= part
        if part == ".." or current.is_symlink():
            raise DogmaError(ErrorCode.CORPUS_PATH_INVALID, "Corpus paths must stay in the repo.")
    if not path.resolve().is_relative_to(root):
        raise DogmaError(ErrorCode.CORPUS_PATH_INVALID, "Corpus paths must stay in the repo.")
    return path


def connect(settings: Settings, *, create: bool = False) -> sqlite3.Connection:
    """Open and validate the corpus schema; search connections are read-only."""
    path = safe_path(settings, "corpus/index.db")
    if not create and not path.exists():
        raise DogmaError(ErrorCode.CORPUS_INDEX_MISSING, "Run dogma corpus sync first.")
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path if create else f"{path.as_uri()}?mode=ro", uri=not create)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if create and version == 0:
            conn.executescript(_SCHEMA)
        elif version != SCHEMA_VERSION:
            raise sqlite3.DatabaseError("Unsupported corpus schema")
        conn.execute("SELECT source_id, path, digest FROM sources LIMIT 0")
        conn.execute(
            "SELECT id, source_id, path, page, headings, ordinal, text, topic_ids "
            "FROM chunks LIMIT 0"
        )
        conn.execute("SELECT rowid,text,headings FROM chunks_fts LIMIT 0")
        conn.execute("SELECT chunk_id,topic_id FROM chunk_topics LIMIT 0")
        return conn
    except sqlite3.Error:
        conn.close()
        raise


def replace_source(
    conn: sqlite3.Connection, source_id: str, path: str, digest: str, chunks: tuple[Chunk, ...]
) -> None:
    """Replace one source and all its FTS rows in one transaction."""
    with conn:
        conn.execute("DELETE FROM sources WHERE source_id=?", (source_id,))
        conn.execute("INSERT INTO sources VALUES(?,?,?)", (source_id, path, digest))
        for chunk in chunks:
            cursor = conn.execute(
                "INSERT INTO chunks(source_id,path,page,headings,ordinal,text,topic_ids) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    chunk.source_id,
                    chunk.path,
                    chunk.page,
                    json.dumps(chunk.headings),
                    chunk.ordinal,
                    chunk.text,
                    json.dumps(chunk.topic_ids),
                ),
            )
            conn.executemany(
                "INSERT INTO chunk_topics VALUES(?,?)",
                [(cursor.lastrowid, topic) for topic in chunk.topic_ids],
            )


def search(
    settings: Settings, query: str, *, topics: tuple[str, ...] = (), k: int = 5
) -> tuple[SearchHit, ...]:
    """Retrieve literal terms using AND, with stable BM25/source/ordinal ordering.

    Operators and punctuation cannot change query syntax. Limits are 1..100,
    queries at most 4096 characters; empty-token queries return no hits.
    """
    if type(k) is not int or not 1 <= k <= 100 or len(query) > 4096:
        raise DogmaError(ErrorCode.CORPUS_QUERY_INVALID, "Use a limit of 1-100 and a short query.")
    terms = re.findall(r"[^\W_]+", query, re.UNICODE)
    if not terms:
        return ()
    literal = " AND ".join('"' + term + '"' for term in terms)
    topic_json = json.dumps(topics)
    conn: sqlite3.Connection | None = None
    try:
        conn = connect(settings)
        rows = conn.execute(
            "SELECT c.source_id,c.path,c.page,c.headings,c.ordinal,c.text,c.topic_ids,"
            "bm25(chunks_fts) FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? AND (? = '[]' OR EXISTS("
            "SELECT 1 FROM chunk_topics t WHERE t.chunk_id=c.id AND "
            "t.topic_id IN (SELECT value FROM json_each(?)))) "
            "ORDER BY bm25(chunks_fts),c.source_id,c.ordinal LIMIT ?",
            (literal, topic_json, topic_json, k),
        ).fetchall()
        return tuple(_hit(row) for row in rows)
    except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
        raise DogmaError(
            ErrorCode.CORPUS_INDEX_INVALID,
            "Corpus index is unreadable; rebuild corpus/index.db with corpus sync.",
        ) from exc
    finally:
        if conn is not None:
            conn.close()


def _hit(row: tuple[str, str, int | None, str, int, str, str, float]) -> SearchHit:
    source_id, path, page, headings, ordinal, text, topics, score = row
    chunk = Chunk(
        source_id, path, page, tuple(json.loads(headings)), ordinal, text, tuple(json.loads(topics))
    )
    label = source_id + (f" p.{page}" if page is not None else "")
    label = label.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    return SearchHit(chunk, f"[{label}]({quote(path, safe='/')})", score)
